# load the pt file
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM
from ChatData import ChatData
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.add_special_tokens({"pad_token": "<PAD>",
                              "bos_token": "<START>",
                              "eos_token": "<END>"})

tokenizer.add_tokens(["<bot>:"])
tokenizer.pad_token = "<PAD>"
tokenizer.bos_token = "<START>"
tokenizer.eos_token = "<END>"


device = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"

model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 300
model.config.max_new_tokens = 300


model.load_state_dict(torch.load('./models/best_model.pt'))
model = model.to(device)

model.eval()


def clean_output(text):
    """
    Clean the output text
    """
    text = text.replace("<START>", "")
    text = text.replace("<bot>:", "\nSheldon :")
    text = text.replace("<END>", "")
    text = text.replace("<pad> ", "")
    text = text.replace("<pad>", "")

    return text


def infer(inp, f=0):
    """
    Infer from the model
    """
    # model.eval()
    inp = "<START> "+inp+"<bot>: "
    inp = tokenizer(inp, return_tensors="pt")

    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)

    output = model.generate(X, attention_mask=a)
    output = tokenizer.decode(output[0])

    if (f):
        index = output.find("<bot>:")
        output = output[index:]
    # model.train()
    return clean_output(output)



while True:
    inp = input("You: ")
    if inp == "exit":
        break
    print(infer(inp,1))
    
    
