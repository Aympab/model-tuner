import onnx
import numpy as np
import tvm.relay as relay
import tvm
from tvm.contrib import graph_executor
import os
import timeit
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
import sys

nb_th=tvm.runtime.num_threads()


logfolder='logs'
models_root_path = "models" #Where are stored the onnx models
input_name = "input_1:0"     #The input name of the ONNX model

#TODO : input_shape
#TODO : output_shape


batch_sizes = [2**x for x in range(0,10)]


target='llvm -libs=blas,mkl -mcpu=cascadelake -opt-level=3 -fast-math -num-cores=24' 

number = 25 #Number of time to run the model to have an average, is 1 repeat
repeat = 3  #How many measurement of each config we take
min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
timeout = 20  # in seconds
trials = 1500 #set to 1500 for CPU and 3000 for GPU
early_stopping = int(trials*0.8)
opt=4 #Optim level

print(f'AutoTuning NN with :\n\
target : {target}\n\
number (of different configuration to test) : {number}\n\
repeat (how many measurement per configuration) : {repeat}\n\
trials : {trials}\n\
early_stopping : {early_stopping}\n\
\nLOG_FOLDER : {logfolder}\n\
TVM_NUM_THREADS : {nb_th}\n')




for bs in batch_sizes :
    print(f"\n###############################################################\n\
                AUTOTUNING FOR BATCH SIZE {bs} \n\
###############################################################\n")
# bs=3000
    try:
        #----------------------------------------------------------------------#
        #----------------------------------------------------------------------#
        #                              ONNX LOADING                            #
        #----------------------------------------------------------------------#
        #----------------------------------------------------------------------#
        model_path=os.path.join(models_root_path, "model_bs" + str(bs) + ".onnx")
        onnx_model = onnx.load(model_path)
        # np.random.seed(0)


        #----------------------------------------------------------------------#
        #----------------------------------------------------------------------#
        #                            COMPILE UNOPT MODEL                       #
        #----------------------------------------------------------------------#
        #----------------------------------------------------------------------#

        input = np.random.rand(bs,1,2)
        shape_dict = {input_name: input.shape}

        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

        with tvm.transform.PassContext(opt_level=opt):
            lib = relay.build(mod, target=target, params=params)

        dev = tvm.device(str(target), 0)
        module = graph_executor.GraphModule(lib["default"](dev))


        #----------------------------------------------------------------------#
        #----------------------------------------------------------------------#
        #                          EXECUTE UNOPT MODEL                         #
        #----------------------------------------------------------------------#
        #----------------------------------------------------------------------#
        dtype = "float32"
        module.set_input(input_name, input)
        # print(module.get_input_info())
        module.run()
        output_shape = (bs, 1, 8)
        tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

        # print(tvm_output)
        #----------------------------------------------------------------------#
        #----------------------------------------------------------------------#
        #                              BASIC METRICS                           #
        #----------------------------------------------------------------------#
        #----------------------------------------------------------------------#
        timing_number = 10
        timing_repeat = 10
        unoptimized = (
            np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
            * 1000
            / timing_number
        )
        unoptimized = {
            "mean": np.mean(unoptimized),
            "median": np.median(unoptimized),
            "std": np.std(unoptimized),
        }
        print(unoptimized)


        #----------------------------------------------------------------------#
        #----------------------------------------------------------------------#
        #                             AUTOTUNE MODEL                           #
        #----------------------------------------------------------------------#
        #----------------------------------------------------------------------#
        print('Autotuning...')

        # create a TVM runner
        runner = autotvm.LocalRunner(
            number=number,
            repeat=repeat,
            timeout=timeout,
            min_repeat_ms=min_repeat_ms,
            enable_cpu_cache_flush=True,
        )

        #Change number of trials to 1500 on CPU and 3000-4000 on GPU
        tuning_option = {
            "tuner": "xgb",
            "trials": trials,
            "early_stopping": early_stopping,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func="default"), runner=runner
            ),
            "tuning_records": os.path.join(logfolder,"neuralnet-bs"+ str(bs)+"-autotuning.json"),
        }


        # begin by extracting the tasks from the onnx model
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

        # Tune the extracted tasks sequentially.
        for i, task in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            tuner_obj = XGBTuner(task, loss_type="rank")
            tuner_obj.tune(
                n_trial=min(tuning_option["trials"], len(task.config_space)),
                early_stopping=tuning_option["early_stopping"],
                measure_option=tuning_option["measure_option"],
                callbacks=[
                    autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                    autotvm.callback.log_to_file(tuning_option["tuning_records"]),
                ],
            )


        #----------------------------------------------------------------------#
        #----------------------------------------------------------------------#
        #                             COMPARE MODELS                           #
        #----------------------------------------------------------------------#
        #----------------------------------------------------------------------#
        with autotvm.apply_history_best(tuning_option["tuning_records"]):
            with tvm.transform.PassContext(opt_level=opt, config={}):
                lib = relay.build(mod, target=target, params=params)
                lib.export_library(os.path.join(logfolder,"compiledNN-bs" + str(bs) + ".so"))

                print(f"Exported library with batch size {bs}")



                dev = tvm.device(str(target), 0)
                module = graph_executor.GraphModule(lib["default"](dev))


                timing_number = 10
                timing_repeat = 10
                optimized = (
                    np.array(timeit.Timer(lambda: module.run()).repeat(repeat=timing_repeat, number=timing_number))
                    * 1000
                    / timing_number
                )
                optimized = {"mean": np.mean(optimized), "median": np.median(optimized), "std": np.std(optimized)}


                print("optimized: %s" % (optimized))
                print("unoptimized: %s" % (unoptimized))
                print(f"Average speedup : {unoptimized['mean']/optimized['mean']}")
    
    except:
        print(f"ERROR WHEN TRYING WITH BATCH SIZE {bs}", file=sys.stderr)
        pass