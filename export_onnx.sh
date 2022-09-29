#!/bin/bash

#TODO : change the --saved_model path to you Tensorflow model location
#TODO : change the --inputs name to the right name and input size

for i in {0..0}
do
    bs=$((2**${i}))
    echo "Exporting for bs ${bs}"
    python -m tf2onnx.convert --saved-model models/saved-toy \
                              --output models/onnx/model_bs${bs}.onnx \
                              --inputs Conv1_input:0[${bs},28,28,1]
done
