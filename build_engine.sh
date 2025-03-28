export LD_LIBRARY_PATH=./LayerNormPlugin/:$LD_LIBRARY_PATH
python trt_builder/builder.py -x bert-base-uncased/model.onnx -c bert-base-uncased/ -o bert-base-uncased/model_fp16.plan -f | tee log.txt
