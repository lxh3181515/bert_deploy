import onnxruntime
import onnxsim
import onnx
import time
import os
from transformers import BertTokenizer

BERT_PATH = './bert-base-uncased'
PROVIDER = 'CUDAExecutionProvider'


def export_sim(in_path, out_path):
    onnx_model = onnx.load(in_path)
    try:
        model_sim, check = onnxsim.simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
    except Exception as e:
        print(f'Simplifier failure: {e}')
    
    onnx.save(model_sim, out_path)
    print("Simplify model is saved as ", out_path)


def load_session():
    print("===================load session=======================")
    
    onnx_path = os.path.join(BERT_PATH, "model.onnx")
    onnx_sim_path = os.path.join(BERT_PATH, "model-sim.onnx")
    
    # If simplified model already exists, then use it.
    if not os.path.exists(onnx_sim_path):
        if not os.path.exists(onnx_path):
            print("ONNX file not exist!")
            return None
        export_sim(onnx_sim_path)
    
    # Check if GPU is avalidable.
    available_providers = onnxruntime.get_available_providers()
    if not PROVIDER in available_providers:
        print("Available providers list:", available_providers)
        print(PROVIDER, " invalid!")
        return None
    providers = [PROVIDER]
    onnx_session = onnxruntime.InferenceSession(onnx_sim_path, providers=providers)
    return onnx_session


def infer(model, tokenizer, text):
    print("===================infer=======================")
    encoded_input = tokenizer.encode_plus(text, return_tensors = "pt")    
    model_input = tuple(encoded_input.values())
    onnx_input = {k.name : v.numpy() for k, v in zip(model.get_inputs(), model_input)}
    print(onnx_input)
    print(onnx_input['input_ids'].dtype)
    ts = time.time()
    for i in range(100):
        outputs = model.run(None, onnx_input)
    print("Average time cost:", round(((time.time() - ts) / 100.0) * 1000, 3), " ms")
    print("Output tensor: ", outputs[0].flatten()[:16], "...")


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."
    
    onnx_session = load_session()
    if onnx_session:
        infer(onnx_session, tokenizer, text)
    