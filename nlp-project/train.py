import torch.nn as nn
import torch.nn.functional as F
import torch
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm 
from configs import *

BCE_loss = nn.BCELoss()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
bert_model = SentenceTransformer(bert_model_name).to(DEVICE)


class NN(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.model = nn.Sequential(
          nn.Linear(embedding_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 1),
          nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
        

if __name__ == "__main__":
    model = NN(EMBEDDING_DIM).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    sentences = [
        "TEST TEST TEST",
        "Another sentence"
    ]
    sentence_embeddings = bert_model.encode(sentences)
    print(sentence_embeddings.shape)
    out = model(torch.tensor(sentence_embeddings))
    out = BCE_loss(out, torch.tensor([1, 0], dtype=torch.float).view(2,1))
    print(out)

    dataset = load_dataset("sst2")
    train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle = True)
    
    for epoch in range(EPOCH):
        print("Epoch ", epoch, ":")
        model.train()
        train_loss = 0
        for input_batch in tqdm(train_loader):
            idx = input_batch["idx"]
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
            print(f'Loss = {loss.item()}')
            print(torch.cuda.mem_get_info())
        break
