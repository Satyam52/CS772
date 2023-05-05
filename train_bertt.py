import torch.nn as nn
import torch.nn.functional as F
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm 
from configs import *
import time
from sklearn.metrics import accuracy_score
from transformers import activations
from bert import BertModel
from transformers import BertTokenizer, BertConfig
from activations.prelu import *
from activations.silu import LearnedSiLU
from activations.swish import LearnedSwish
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

BCE_loss = nn.BCELoss()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.to(DEVICE)




class NN(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.model = nn.Sequential(
          nn.Linear(embedding_dim, 256),
          #LearnedSiLU(),
          LearnedSiLU(embedding_dim=256),
          nn.Linear(256, 1),
          nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    

class DualForward(nn.Module):
    def __init__(self, in_dim = 512, out_dim = 64, activation = nn.ReLU()):
        super().__init__()
        self.layer_1 = nn.Linear(in_dim, out_dim)
        self.layer_2 = nn.Linear(in_dim, out_dim)
        self.activation = activation

    def forward(self, x):
        return self.activation(self.layer_1(x)) * self.activation(self.layer_2(x))
        

def train_sst():
    print("Training SST")
    highest_acc = 0
    for epoch in range(EPOCH):
        #print("Epoch ", epoch, ":")
        model.train()
        train_loss = 0
        dfs = {}
        for name, param in bert_model.named_parameters():
            if name.find('slope') != -1:
                dfs[name] = pd.DataFrame()
        for i, input_batch in tqdm(enumerate(train_loader)):
            sentences = input_batch["sentence"]
            label = input_batch["label"]
            label = label.view(label.shape[0],1).to(torch.float).to(DEVICE)
            optimizer.zero_grad()
            bert_model.zero_grad()
            #sentence_embeddings = bert_model.encode(sentences)
            encoded_input = bert_tokenizer(sentences,padding=True, max_length=128, truncation=True, return_tensors='pt').to(DEVICE)
            output = bert_model(**encoded_input)
            sentence_embeddings = output.last_hidden_state.mean(dim=1)
            out = model(sentence_embeddings)
            loss = BCE_loss(out, label)
            loss.backward()
            # for name, param in bert_model.named_parameters():
            #     if name.find('slope') != -1:
            #         print(name, param.data, param.grad, param.requires_grad)
            optimizer.step()
            train_loss += loss.item()
            if i % 20 == 0:
                for name, param in bert_model.named_parameters():
                    if name.find('slope') != -1:
                        dfs[name] = dfs[name].append(pd.DataFrame(param.data.cpu().numpy().reshape(1,-1)))
        for name, param in bert_model.named_parameters():
            if name.find('slope') != -1:
                dfs[name].to_csv("results/sst/"+name+".csv", index=False)
        y_pred = []
        y_test = []
        with torch.no_grad():
            val_loss = 0
            for input_batch in (val_loader):
                sentences = input_batch["sentence"]
                label = input_batch["label"]
                label = label.view(label.shape[0],1).to(torch.float).to(DEVICE)
                #sentence_embeddings = bert_model.encode(sentences)
                encoded_input = bert_tokenizer(sentences, padding=True, max_length=128, truncation=True, return_tensors='pt').to(DEVICE)
                sentence_embeddings = bert_model(**encoded_input)
                sentence_embeddings = sentence_embeddings.last_hidden_state.mean(dim=1)
                out = model(sentence_embeddings)
                loss = BCE_loss(out, label)
                #print(type(out))
                y_pred.extend(out.squeeze().to('cpu').numpy())
                y_test.extend(label.squeeze().to('cpu').numpy())
                val_loss += loss.item()
        y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        curr_acc = accuracy_score(y_test, y_pred, normalize=True)
        if curr_acc > highest_acc:
            highest_acc = curr_acc
            torch.save(model.state_dict(), "sst_model.pt")
            torch.save(bert_model.state_dict(), "sst_bert_model.pt")
            print("Model Saved")
        y_pred = []
        y_test = []
        with torch.no_grad():
            for input_batch in (train_loader):
                sentences = input_batch["sentence"]
                label = input_batch["label"]
                label = label.view(label.shape[0],1).to(torch.float).to(DEVICE)
                #sentence_embeddings = bert_model.encode(sentences)
                encoded_input = bert_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
                sentence_embeddings = bert_model(**encoded_input)
                sentence_embeddings = sentence_embeddings.last_hidden_state.mean(dim=1)
                out = model(sentence_embeddings)
                y_pred.extend(out.squeeze().to('cpu').numpy())
                y_test.extend(label.squeeze().to('cpu').numpy())
        y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        train_acc = accuracy_score(y_test, y_pred, normalize=True)
        print('Train Acc = ', train_acc)
        print("train loss", train_loss/len(train_loader))
        print(f'Val Acc = {curr_acc}')
        print("val loss", val_loss/len(val_loader))
        print(f'slope = {model.model[1].slope}')
        with open('ss_logs.csv', 'a') as f:
            f.write(f'{train_loss/len(train_loader)},{val_loss/len(val_loader)},{train_acc},{curr_acc},{model.model[1].slope.data}\n')
        exit()



def train_cola():
    print("Training COLA")
    highest_acc = 0
    for epoch in tqdm(range(EPOCH)):
        #print("Epoch ", epoch, ":")
        model.train()
        train_loss = 0
        for input_batch in (train_loader):
            sentences = input_batch["sentence"]
            label = input_batch["label"]
            label = label.view(label.shape[0],1).to(torch.float).to(DEVICE)
            optimizer.zero_grad()
            bert_model.zero_grad()
            encoded_input = bert_tokenizer(sentences,padding=True, max_length=128, truncation=True, return_tensors='pt').to(DEVICE)
            output = bert_model(**encoded_input)
            sentence_embeddings = output.last_hidden_state.mean(dim=1)
            out = model(sentence_embeddings)
            loss = BCE_loss(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        y_pred = []
        y_test = []
        with torch.no_grad():
            val_loss = 0
            for input_batch in (val_loader):
                sentences = input_batch["sentence"]
                label = input_batch["label"]
                label = label.view(label.shape[0],1).to(torch.float).to(DEVICE)
                encoded_input = bert_tokenizer(sentences,padding=True, truncation=True, return_tensors='pt').to(DEVICE)
                output = bert_model(**encoded_input)
                sentence_embeddings = output.last_hidden_state.mean(dim=1)
                out = model(sentence_embeddings)
                y_pred.extend(out.squeeze().to('cpu').numpy())
                y_test.extend(label.squeeze().to('cpu').numpy())
                val_loss += loss.item()
        y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        curr_acc = accuracy_score(y_test, y_pred, normalize=True)
        if curr_acc > highest_acc:
            highest_acc = curr_acc
            torch.save(model.state_dict(), "cola_model.pt")
            print("Model Saved")
        y_pred = []
        y_test = []
        with torch.no_grad():
            for input_batch in (train_loader):
                sentences = input_batch["sentence"]
                label = input_batch["label"]
                label = label.view(label.shape[0],1).to(torch.float).to(DEVICE)
                encoded_input = bert_tokenizer(sentences,padding=True, truncation=True, return_tensors='pt').to(DEVICE)
                output = bert_model(**encoded_input)
                sentence_embeddings = output.last_hidden_state.mean(dim=1)
                out = model(sentence_embeddings)
                loss = BCE_loss(out, label)
                y_pred.extend(out.squeeze().to('cpu').numpy())
                y_test.extend(label.squeeze().to('cpu').numpy())
        y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        train_acc = accuracy_score(y_test, y_pred, normalize=True)
        print('Train Acc = ', train_acc)
        print("train loss", train_loss/len(train_loader))
        print(f'Val Acc = {curr_acc}')
        print("val loss", val_loss/len(val_loader))
        print(f'slope = {model.model[1].slope}')
        with open('cola_logs.csv', 'a') as f:
            f.write(f'{train_loss/len(train_loader)},{val_loss/len(val_loader)},{train_acc},{curr_acc},{model.model[1].slope.data}\n')
    
def train_quora():
    print("Training QUORA")
    highest_acc = 0
    for epoch in tqdm(range(EPOCH)):
        #print("Epoch ", epoch, ":")
        model.train()
        train_loss = 0
        for input_batch in (train_loader):
            questions = input_batch["questions"]
            is_duplicate = input_batch["is_duplicate"]
            sentence_1 = list(questions['text'][0])
            sentence_2 = list(questions['text'][1])
            is_duplicate = is_duplicate.view(is_duplicate.shape[0],1).to(torch.float).to(DEVICE)
            optimizer.zero_grad()
            bert_model.zero_grad()
            encoded_input_1 = bert_tokenizer(sentence_1,padding=True, truncation=True, return_tensors='pt').to(DEVICE)
            encoded_input_2 = bert_tokenizer(sentence_2,padding=True, truncation=True, return_tensors='pt').to(DEVICE)
            output_1 = bert_model(**encoded_input_1)
            output_2 = bert_model(**encoded_input_2)
            sentence_1_embeddings = output_1.last_hidden_state.mean(dim=1)
            sentence_2_embeddings = output_2.last_hidden_state.mean(dim=1)
            concat = torch.cat((sentence_1_embeddings, sentence_2_embeddings), 1)
            out = model(concat)
            loss = BCE_loss(out, is_duplicate)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        y_pred = []
        y_test = []
        with torch.no_grad():
            val_loss = 0
            for input_batch in (val_loader):
                questions = input_batch["questions"]
                is_duplicate = input_batch["is_duplicate"]
                sentence_1 = list(questions['text'][0])
                sentence_2 = list(questions['text'][1])
                is_duplicate = is_duplicate.view(is_duplicate.shape[0],1).to(torch.float).to(DEVICE)
                optimizer.zero_grad()
                bert_model.zero_grad()
                encoded_input_1 = bert_tokenizer(sentence_1,padding=True, truncation=True, return_tensors='pt').to(DEVICE)
                encoded_input_2 = bert_tokenizer(sentence_2,padding=True, truncation=True, return_tensors='pt').to(DEVICE)
                output_1 = bert_model(**encoded_input_1)
                output_2 = bert_model(**encoded_input_2)
                sentence_1_embeddings = output_1.last_hidden_state.mean(dim=1)
                sentence_2_embeddings = output_2.last_hidden_state.mean(dim=1)
                concat = torch.cat((sentence_1_embeddings, sentence_2_embeddings), 1)
                out = model(concat)
                loss = BCE_loss(out, is_duplicate)
                y_pred.extend(out.squeeze().to('cpu').numpy())
                y_test.extend(is_duplicate.squeeze().to('cpu').numpy())
                val_loss += loss.item()
        y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        curr_acc = accuracy_score(y_test, y_pred, normalize=True)
        if curr_acc > highest_acc:
            highest_acc = curr_acc
            torch.save(model.state_dict(), "quora_model.pt")
            print("Model Saved")
        y_pred = []
        y_test = []
        with torch.no_grad():
            for input_batch in (train_loader):
                questions = input_batch["questions"]
                is_duplicate = input_batch["is_duplicate"]
                sentence_1 = list(questions['text'][0])
                sentence_2 = list(questions['text'][1])
                is_duplicate = is_duplicate.view(is_duplicate.shape[0],1).to(torch.float).to(DEVICE)
                optimizer.zero_grad()
                bert_model.zero_grad()
                encoded_input_1 = bert_tokenizer(sentence_1,padding=True, truncation=True, return_tensors='pt').to(DEVICE)
                encoded_input_2 = bert_tokenizer(sentence_2,padding=True, truncation=True, return_tensors='pt').to(DEVICE)
                output_1 = bert_model(**encoded_input_1)
                output_2 = bert_model(**encoded_input_2)
                sentence_1_embeddings = output_1.last_hidden_state.mean(dim=1)
                sentence_2_embeddings = output_2.last_hidden_state.mean(dim=1)
                concat = torch.cat((sentence_1_embeddings, sentence_2_embeddings), 1)
                out = model(concat)
                y_pred.extend(out.squeeze().to('cpu').numpy())
                y_test.extend(is_duplicate.squeeze().to('cpu').numpy())
        y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        train_acc = accuracy_score(y_test, y_pred, normalize=True)
        print('Train Acc = ', train_acc)
        print("train loss", train_loss/len(train_loader))
        print(f'Val Acc = {curr_acc}')
        print("val loss", val_loss/len(val_loader))
        print(f'slope = {model.model[1].slope}')
        with open('quora_logs.csv', 'a') as f:
            f.write(f'{train_loss/len(train_loader)},{val_loss/len(val_loader)},{train_acc},{curr_acc},{model.model[1].slope.data}\n')


model = NN(EMBEDDING_DIM).to(DEVICE)
#select_act_class(bert_model, LearnedSiLU)
#print(type(model.parameters()))
bert_modify_params = []

for name, param in bert_model.named_parameters():
    param.requires_grad = True
    if name.find('slope') != -1:
        print(name)
        bert_modify_params.append(param)
optimizer = AdamW((n for n in (list(model.parameters())+ bert_modify_params)), lr=1e-3)
train_loader = None
val_loader = None
test_loader = None

def eval_func():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased")
    text = ["Replace me by any text you'd like.", "Replace me by any text you'd like."]
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    out = output.last_hidden_state.mean(dim=1)
    print(out.shape)
    nn_model = NN(embedding_dim=768)
    out = nn_model(out)
    print(out.shape)
    loss = nn.BCELoss()
    calc_loss = loss(out, torch.tensor([1.0,0.0]).view(-1,1))
    print(calc_loss)
    calc_loss.backward()
    print(nn_model.model[1].slope.grad)

    for name, param in model.named_parameters():
        if name.find('slope') != -1:
            print(name, param.grad)


if __name__ == "__main__":
    #eval_func()
    print(f'Eval function passed')
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sst2")
    
    args = parser.parse_args()
    if args.dataset == "sst2":
        dataset = load_dataset("sst2")
        train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle = True)
        val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE)
        test_loader = DataLoader(dataset["test"], batch_size = BATCH_SIZE)
        train_sst()
    elif args.dataset == "cola":
        dataset = load_dataset("glue", "cola")
        train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle = True)
        val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE)
        test_loader = DataLoader(dataset["test"], batch_size = BATCH_SIZE)
        train_cola()
    elif args.dataset == "quora":
        dataset = load_dataset("quora")['train'].train_test_split(test_size=0.2)
        train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle = True)
        val_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE)
        train_quora()
    else:
        print("Invalid dataset")
        exit()
