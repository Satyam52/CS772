import torch.nn as nn
import torch.nn.functional as F
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm 
from configs import *
import time
from sklearn.metrics import accuracy_score
from activations.prelu import *
from activations.relun import *
from activations.silu import LearnedSiLU
import argparse


BCE_loss = nn.BCELoss()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
bert_model = SentenceTransformer(bert_model_name).to(DEVICE)

class NN(nn.Module):
    def __init__(self, embedding_dim=768, activation=None):
        super().__init__()
        if activation == None:
            print("provide a valid activation function")
            exit()
        self.model = nn.Sequential(
          nn.Linear(embedding_dim, 256),
          activation(),
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
        

def train_sst(act_name):
    print("Training SST")
    highest_acc = 0
    for epoch in tqdm(range(EPOCH)):
        print("Epoch ", epoch, ":")
        model.train()
        train_loss = 0
        y_pred = []
        y_test = []
        for input_batch in (train_loader):
            sentences = input_batch["sentence"]
            label = input_batch["label"]
            label = label.view(label.shape[0],1).to(torch.float).to(DEVICE)
            optimizer.zero_grad()
            sentence_embeddings = bert_model.encode(sentences)
            out = model(torch.tensor(sentence_embeddings).to(DEVICE))
            loss = BCE_loss(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            y_pred.extend(out.detach().clone().squeeze().to('cpu').numpy())
            y_test.extend(label.squeeze().to('cpu').numpy())
        y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        train_acc = accuracy_score(y_test, y_pred, normalize=True)
        
        
        y_pred = []
        y_test = []   
        with torch.no_grad():
            val_loss = 0
            for input_batch in (val_loader):
                sentences = input_batch["sentence"]
                label = input_batch["label"]
                label = label.view(label.shape[0],1).to(torch.float).to(DEVICE)
                sentence_embeddings = bert_model.encode(sentences)
                out = model(torch.tensor(sentence_embeddings).to(DEVICE))
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
            print("Model Saved")
        print("train loss", train_loss/len(train_loader))
        print(f'Acc = {curr_acc}')
        print("val loss", val_loss/len(val_loader))
        
        with open("loss/"+"loss_"+name+"_sst2"+".txt", 'a') as f:
            f.write(f'{train_loss/len(train_loader)} {val_loss/len(val_loader)} {train_acc} {curr_acc}\n')
        if name in pact:
            print(f'param = {model.model[1].get_param()}')
            with open("loss/"+"param_"+name+"_sst2"+".txt", 'a') as f:
                f.write(f'{model.model[1].get_param()}\n')


def train_cola(act_name):
    print("Training COLA")
    highest_acc = 0
    cnt = 0
    for epoch in tqdm(range(EPOCH)):
        print("Epoch ", epoch, ":")
        model.train()
        train_loss = 0
        y_pred = []
        y_test = []
        for input_batch in (train_loader):
            sentences = input_batch["sentence"]
            label = input_batch["label"]
            label = label.view(label.shape[0],1).to(torch.float).to(DEVICE)
            optimizer.zero_grad()
            sentence_embeddings = bert_model.encode(sentences)
            out = model(torch.tensor(sentence_embeddings).to(DEVICE))
            loss = BCE_loss(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            y_pred.extend(out.detach().clone().squeeze().to('cpu').numpy())
            y_test.extend(label.squeeze().to('cpu').numpy())
        y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        train_acc = accuracy_score(y_test, y_pred, normalize=True)
        
        y_pred = []
        y_test = []
        with torch.no_grad():
            val_loss = 0
            for input_batch in (val_loader):
                sentences = input_batch["sentence"]
                label = input_batch["label"]
                label = label.view(label.shape[0],1).to(torch.float).to(DEVICE)
                sentence_embeddings = bert_model.encode(sentences)
                out = model(torch.tensor(sentence_embeddings).to(DEVICE))
                loss = BCE_loss(out, label)
                #print(type(out))
                y_pred.extend(out.squeeze().to('cpu').numpy())
                y_test.extend(label.squeeze().to('cpu').numpy())
                val_loss += loss.item()

        y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        curr_acc = accuracy_score(y_test, y_pred, normalize=True)
        if curr_acc > highest_acc:
            highest_acc = curr_acc
            torch.save(model.state_dict(), "cola_model.pt")
            print("Model Saved")
        print("train loss", train_loss/len(train_loader))
        print(f'Acc = {curr_acc}')
        print("val loss", val_loss/len(val_loader))
        
        with open("loss/"+"loss_"+name+"_cola"+".txt", 'a') as f:
            f.write(f'{train_loss/len(train_loader)} {val_loss/len(val_loader)} {train_acc} {curr_acc}\n')
        if name in pact:
            print(f'param = {model.model[1].get_param()}')
            with open("loss/"+"param_"+name+"_cola"+".txt", 'a') as f:
                f.write(f'{model.model[1].get_param()}\n')

def train_quora(act_name):
    highest_acc = 0
    for epoch in tqdm(range(EPOCH)):
        print("Epoch ", epoch, ":")
        model.train()
        train_loss = 0
        y_pred = []
        y_test = []
        for input_batch in tqdm(train_loader):
            questions = input_batch["questions"]
            is_duplicate = input_batch["is_duplicate"]
            sentence_1 = list(questions['text'][0])
            sentence_2 = list(questions['text'][1])
            is_duplicate = is_duplicate.view(is_duplicate.shape[0],1).to(torch.float).to(DEVICE)
            optimizer.zero_grad()
            sentence_1_embeddings = torch.tensor(bert_model.encode(sentence_1))
            sentence_2_embeddings = torch.tensor(bert_model.encode(sentence_2))
            concat = torch.cat((sentence_1_embeddings, sentence_2_embeddings), 1)
            out = model(torch.tensor(concat).to(DEVICE))
            loss = BCE_loss(out, is_duplicate)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            y_pred.extend(out.detach().clone().squeeze().to('cpu').numpy())
            y_test.extend(is_duplicate.squeeze().to('cpu').numpy())
        
        y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        train_acc = accuracy_score(y_test, y_pred, normalize=True)
        print("train loss", train_loss/len(train_loader))
        y_pred = []
        y_test = []
        cnt = 0
        with torch.no_grad():
            val_loss = 0
            for input_batch in (val_loader):
                questions = input_batch["questions"]
                is_duplicate = input_batch["is_duplicate"]
                sentence_1 = list(questions['text'][0])
                sentence_2 = list(questions['text'][1])
                is_duplicate = is_duplicate.view(is_duplicate.shape[0],1).to(torch.float).to(DEVICE)
                optimizer.zero_grad()
                sentence_1_embeddings = torch.tensor(bert_model.encode(sentence_1))
                sentence_2_embeddings = torch.tensor(bert_model.encode(sentence_2))
                concat = torch.cat((sentence_1_embeddings, sentence_2_embeddings), 1)
                out = model(torch.tensor(concat).to(DEVICE))
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
        print(f'Acc = {curr_acc}')
        print("val loss", val_loss/len(val_loader))
        
        with open("loss/"+"loss_"+name+"_quora"+".txt", 'a') as f:
            f.write(f'{train_loss/len(train_loader)} {val_loss/len(val_loader)} {train_acc} {curr_acc}\n')
        if name in pact:
            print(f'param = {model.model[1].get_param()}')
            with open("loss/"+"param_"+name+"_quora"+".txt", 'a') as f:
                f.write(f'{model.model[1].get_param()}\n')

model = None
optimizer = None
train_loader = None
val_loader = None
test_loader = None

pact = ["prelu", "relun"]

activations = {
            #    "prelu" : PReLU,
            #    "relun": ReLUN,
               "relu": nn.ReLU,
            #    "gelu": nn.GELU,
            #    "sigmoid": nn.Sigmoid,
            #    "tanh": nn.Tanh,
            #    "leakyr": nn.LeakyReLU,
            #    "swish": Swish,
            "learnedSilu":LearnedSiLU

            }

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset", type=str, default="cola")
    # args = parser.parse_args()

    for name, activation in activations.items():
        
        if True or dataset_name == "sst2":
            model = NN(EMBEDDING_DIM, activation=activation).to(DEVICE)
            optimizer = AdamW(model.parameters(), lr=1e-3)
            dataset = load_dataset("sst2")
            train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle = True)
            val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE)
            test_loader = DataLoader(dataset["test"], batch_size = BATCH_SIZE)
            train_sst(name)
        # if True or args.dataset == "cola":
        #     model = NN(EMBEDDING_DIM, activation=activation).to(DEVICE)
        #     optimizer = AdamW(model.parameters(), lr=1e-3)
        #     dataset = load_dataset("glue", "cola")
        #     train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle = True)
        #     val_loader = DataLoader(dataset["validation"], batch_size=BATCH_SIZE)
        #     test_loader = DataLoader(dataset["test"], batch_size = BATCH_SIZE)
        #     train_cola(name)
        # if True or args.dataset == "quora":
        #     model = NN(EMBEDDING_DIM*2, activation=activation).to(DEVICE)
        #     optimizer = AdamW(model.parameters(), lr=1e-3)
        #     dataset = load_dataset("quora")['train'].train_test_split(test_size=0.2)
        #     train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle = True)
        #     val_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE)
        #     train_quora(name)
        else:
            print("Invalid dataset")
            exit()
