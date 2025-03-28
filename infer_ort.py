import onnxruntime
import onnxsim
import onnx
import time
import os
from transformers import BertTokenizer
import torch
from torch.nn import functional as F

BERT_PATH = './bert-base-uncased'
PROVIDER = 'CUDAExecutionProvider'


def load_session():
    print("===================load session=======================")
    
    onnx_sim_path = os.path.join(BERT_PATH, "model_sim.onnx")
    
    # If simplified model already exists, then use it.
    if not os.path.exists(onnx_sim_path):
        print("ONNX file not exist!")
        return None
    
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
    mask_index = torch.where(encoded_input["input_ids"][0] == tokenizer.mask_token_id)
    
    onnx_input = {k: v.numpy() for k, v in encoded_input.items()}
    print(onnx_input)
    ts = time.time()
    for i in range(100):
        outputs = model.run(None, onnx_input)
    print("Average time cost:", round(((time.time() - ts) / 100.0) * 1000, 3), " ms")
    print("output shape:", outputs[0].shape)
    print("output:", outputs[0])
    print("output sum:", outputs[0].sum())
    
    logits = torch.Tensor(outputs[0]).cuda()
    mask_word = logits[0, mask_index, :]
    mask_word = F.softmax(mask_word, dim = 1)
    top_10 = torch.topk(mask_word, 10, dim = 1)[1][0]
    print(top_10)
    print("model test topk10 output:")
    for token in top_10:
        word = tokenizer.decode([token])
        new_sentence = text.replace(tokenizer.mask_token, word)
        print(token, new_sentence)


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    text = "The capital of France, " + tokenizer.mask_token + ", contains the Eiffel Tower."
    
    onnx_session = load_session()
    if onnx_session:
        infer(onnx_session, tokenizer, text)
    