import onnx
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import os
import timeit
from tvm.runtime import profiler_vm
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
from tvm.relay.testing import mlp

target='llvm -libs=cblas,mkl -mcpu=cascadelake -opt-level=3 -fast-math' 
target='llvm'
number = 10 #Number of time to run the model to have an average, is 1 repeat
repeat = 5 #How many measurement of each config we take
min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
timeout = 20  # in seconds
trials = 1500 #set to 1500 for CPU and 3000 for GPU
early_stopping = (trials//2)+1
opt=4 #Optim level

bs=3536


logfolder="logs/"+str(bs)+"/"

#Load onnx model
# models_root_path = "models/"
# model_path=os.path.join(models_root_path, "model_bs" + str(bs) + ".onnx")
# onnx_model = onnx.load(model_path)
# input_name = "input_1:0"

# input = np.random.rand(bs,1,2).astype("float32")
# shape_dict = {input_name: input.shape}

# mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with tvm.transform.PassContext(opt_level=opt):
    lib = relay.build(mod, target=target, params=params)

dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
runner = autotvm.LocalRunner(
    number=number,
    repeat=repeat,
    timeout=timeout,
    min_repeat_ms=min_repeat_ms,
    enable_cpu_cache_flush=True,
)

# tuning_option = {
#     "tuner": "xgb",
#     "trials": trials,
#     "early_stopping": early_stopping,
#     "measure_option": autotvm.measure_option(
#         builder=autotvm.LocalBuilder(build_func="default"), runner=runner
#     ),
#     "tuning_records": logfolder+"samponet-bs"+ str(bs)+"-autotuning.json",
# }

with autotvm.apply_history_best(logfolder+"samponet-bs"+ str(bs)+"-autotuning.json"):
    with tvm.transform.PassContext(opt_level=opt, config={}):
        lib = relay.build(mod, target=target, params=params)
        # lib.export_library("compiledNN-bs" + str(bs) + ".so")
        # print(f"Exported library with batch size {bs}")

        dev = tvm.device(str(target), 0)
        module = graph_executor.GraphModule(lib["default"](dev))


        # dev = tvm.cpu()
        # mod, params = mlp.get_workload(bs, image_shape=(1,2))


        # from tvm.relay import transform
        # mod = transform.RemoveUnusedFunctions()(mod)
        # mod = transform.AnnotateTarget("dnnl")(mod)


        exe = relay.vm.compile(mod, target, params=params)
        vm = profiler_vm.VirtualMachineProfiler(exe, dev)

        data = tvm.nd.array(np.random.rand(bs, 1, 2).astype("float32"), device=dev)

        # print(mod.astext(show_meta_data=False))
        # print(vm)
        # vm.profile(data)
        report = vm.profile(
            [data],
            func_name="main",
            collectors=[tvm.runtime.profiling.PAPIMetricCollector()],
        )
        # print(report)