from transformers import GPT2LMHeadModel, GPT2Tokenizer, set_seed
from transformers import Trainer, TrainingArguments
from ChatData import ChatData
from torch.optim import Adam, Adadelta, Adamax, AdamW
from transformers.optimization import Adafactor, AdafactorSchedule, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader


from tqdm import tqdm
import torch
import random
import wandb
import pandas as pd


FILENAME = "./Dataset/sheldon_chats.json"
random_seed_number = random.randint(0, 200)

# epochs will be changed and then fixed once an approximation is found as to where the model starts overfitting

config = {
    "learning_rate": 8e-5,
    "batch_size": 4,
    "project_name": "iNLP_Project",
    "entity_name": "mhardik003",
    "random_seed": random_seed_number,
}


# get a random number generator between 0 and 100
print("Seed  : ", random_seed_number)
# always good to set a fixed seed for reproducibility
set_seed(random_seed_number)


def init_wandb(selected_model, config):
    wandb.init(project=config["project_name"], entity=config["entity_name"], config=config,
               name=selected_model + "_" + str(config["random_seed"]))


def clean_output(text):
    """
    Clean the output text
    """
    text = text.replace("<START>", "Question :")
    text = text.replace("<bot>:", "\nSheldon :")
    text = text.replace("<END>", "")
    text = text.replace("<pad> ", "")
    text = text.replace("<pad>", "")

    return text


def train(chatData, model, optim, scheduler, NUM_EPOCHS=10):
    """
    This function trains the model and is built for reproducibility

    Arguements 
    chatData : The dataset
    model : The model to be trained
    optim : The optimizer used for training
    scheduler : The scheduler used for training
    """
    model.train()
    answers = []
    total_loss = 0
    avg_loss = 0
    try:
        for i in tqdm(range(NUM_EPOCHS)):
            # change learning rate
            # if (i % 3 == 0 and optim.param_groups[0]['lr'] > 1e-5):
            #     optim.param_groups[0]['lr'] /= 2

            for X, a in tqdm(chatData):
                model.zero_grad()
                optim.zero_grad()
                X = X.to(device)
                a = a.to(device)
                # calculate the loss of the model
                loss = model(X, attention_mask=a, labels=X)[0]
                total_loss += loss.item()
                loss.backward()
                optim.step()
                scheduler.step()

            avg_loss = total_loss / (len(chatData)*(i+1))
            ans_ques1 = infer("Hello, how are you?", 1)
            ans_ques2 = infer("What is your name?", 1)
            ans_ques3 = infer("Is your name Sheldon? Yes or No?", 1)
            ans_ques4 = infer("Who are your friends?", 1)
            ans_ques5 = infer("Where do you work at?", 1)
            ans_ques6 = infer(
                "What inspired you to pursue a career in physics?", 1)
            ans_ques7 = infer("What is your girlfriend's name?", 1)
            ans_ques8 = infer("What does your girlfriend do?", 1)
            model_name = "model_state_" + str(random_seed_number) + ".pt"
            torch.save(model.state_dict(), "./models/" + model_name)
            answers.append([ans_ques1, ans_ques2, ans_ques3, ans_ques4,
                           ans_ques5, ans_ques6, ans_ques7, ans_ques8])
            question_answers = pd.DataFrame(
                answers, columns=["Hello, how are you?", "What is your name?", "Is your name Sheldon? Yes or No?", "Who are your friends?", "Where do you work at?", "What inspired you to pursue a career in physics?", "What is your girlfriend's name?", "What does your girlfriend do?"])
            wandb_table = wandb.Table(dataframe=question_answers)

            wandb.log({"loss": loss, "Average Loss": avg_loss, "epoch": i,
                       "learning_rate": optim.param_groups[0]['lr'], "questions_answers": wandb_table})

            print("-"*100)
            print("Question : Hello, how are you?", ans_ques1)
            print("Question : What is your name?", ans_ques2)
            print("Question : Is your name Sheldon? Yes or No?", ans_ques3)
            print("Question : Who are your friends?", ans_ques4)
            print("Question : Where do you work at?", ans_ques5)
            print(
                "Question : What inspired you to pursue a career in physics?", ans_ques6)
            print("Question : What is your girlfriend's name?", ans_ques7)
            print("Question : What does your girlfriend do?", ans_ques8)

            # with open("output.txt", 'a') as f:

            #     f.write("\n" + "-"*100 + "\n")
            #     f.write(ans_ques1 + "\n")
            #     f.write(ans_ques2 + "\n")
            #     f.write(ans_ques3 + "\n")
            model.eval()

    except KeyboardInterrupt:
        model.eval()
        pass


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


# check if cuda is available
device = "cuda" if torch.cuda.is_available(
) else "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device : ", device)

NUM_EPOCHS = int(input("Enter number of epochs : "))

selected_model = "GPT2Medium"

model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained(
    "gpt2-medium", pad_token="<pad>", bos_token="<START>", eos_token="<END>")
# config = model.config

init_wandb(selected_model, config)

# training_args = TrainingArguments(
#     output_dir="./results",  # output directory
#     num_train_epochs=NUM_EPOCHS,  # total number of training epochs
#     per_device_train_batch_size=1,  # batch size per device during training
#     per_device_eval_batch_size=1,  # batch size for evaluation
#     warmup_steps=500,  # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,  # strength of weight decay
#     logging_dir="./logs",  # directory for storing logs
#     logging_steps=10,
#     save_steps=1000,
#     save_total_limit=1,
#     evaluation_strategy="steps",
#     load_best_model_at_end=True,
#     metric_for_best_model="loss",
#     greater_is_better=False,
#     report_to="wandb",
#     run_name="GPT2" if model_type == "1" else "GPTNeo",

# )

# trainer = Trainer(
#     model=model,  # the instantiated 🤗 Transformers model to be trained
#     args=training_args,  # training arguments, defined above
#     train_dataset=chatData,  # training dataset
#     # eval_dataset=chatData,  # evaluation dataset
#     tokenizer=tokenizer,

# )


# add special tokens
tokenizer.add_tokens(["<bot>:"])

model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 30
model.config.use_cache = True  # for faster generation of text
# model.config.repetition_penalty = 0.75 # how much to penalize for repeating words
# model.config.temperature = 0.85 # creativity setting (1 means most creative/random)
model.config.max_new_tokens = 100
model.config.attention_layers = ["global"] * 12

# freeze the first 6 layers of the model
for param in model.parameters():
    param.requires_grad = False

# only the last 6 layers will be trained
for param in model.transformer.h[6:].parameters():
    param.requires_grad = True


model.to(device)


chatData = ChatData(FILENAME, tokenizer)
chatData = DataLoader(chatData, batch_size=config["batch_size"], shuffle=True)

model.train()

optim = AdamW(model.parameters(),
              lr=config["learning_rate"], eps=1e-8, weight_decay=0.01)
# add scheduler to the optimizer
scheduler = get_linear_schedule_with_warmup(
    optim, num_warmup_steps=0, num_training_steps=len(chatData)*NUM_EPOCHS)

wandb.watch(model)


print("training .... ")
train(chatData, model, optim, scheduler, NUM_EPOCHS)

print("infer from model : ")
while True:
    inp = input("You :")
    print(infer(inp, 1))
