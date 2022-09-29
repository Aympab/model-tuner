# Model inference tuner with AutoTVM
This repo is a model compiler and tuner using [TVM](https://tvm.apache.org/).

### Various operations can be done :
* Transform a TensorFlow saved model to ONNX and fix an input size
* Compile (multiple) model(s) as `.so` dynamic libraries
* Optimize the throughput of models using
[AutoTVM](https://tvm.apache.org/docs/reference/api/python/autotvm.html)
* Benchmark the compiled models


---
## Requirements
### To compile and auto-tune a model :
1. TVM compiled with the right flags (see website TVM) and installed

2. Python environment with TVM, onnx, numpy and xgboost

3. ML Model, either :
    * To protocol buffer (i.e. `.pb`) format, saved with Tensorflow
        * If you have a `.pb` saved model, you will need to install
        [`tf2onnx`](https://github.com/onnx/tensorflow-onnx) as well
        
    * To ONNX format, with a fixed batch size for the input. If you already have
    an ONNX model, you don't need `tf2onnx`

4. Your hardware architecture details
5. Any external libraries required by your hardware (MKL, CUDA, etc.). Compile
TVM with the associated flags on


### To profile the models
1. The [PAPI Timers](https://bitbucket.org/icl/papi/src/master/) library

2. TVM compiled with the `USE_PAPI` flag. You can follow this
[tutorial](https://tvm.apache.org/docs/how_to/profile/papi.html) for the setup

3. A python environment with numpy, matplotlib

## How to step by step
1. Convert your TF model to ONNX format with the right batch size
    * Use Netron to find the input name of your model
    * Use the `export_onnx.sh` script with the right batch size
2. Run the AutoTuner, see TVM tutorial
    * Tune your options :
        * target : hardware ...
        * min repeat ...
3. Done ! You can use your moodel in C++ with the
[TVM Runtime](https://tvm.apache.org/docs/how_to/deploy/cpp_deploy.html). Or
anywhere you want.


## Benchmark different models
#TODO

Use the benchmark script on the output to parse and plot