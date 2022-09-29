#!/bin/bash

#TODO : change the --saved_model path to you Tensorflow model location
#TODO : change the --inputs name to the right name and input size

for i in {0..10}
do
    bs=$((2**${i}))
    echo "Exporting for bs ${bs}"
    python -m tf2onnx.convert --saved-model path/to/my/saved/model/folder/.../ \
                              --output model_bs${bs}.onnx \
                              --inputs input_1:0[${bs},1,2] 
done
