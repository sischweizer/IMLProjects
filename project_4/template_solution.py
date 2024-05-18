# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from transformers import DistilBertModel
from transformers import DistilBertTokenizer
from transformers import AutoTokenizer, AlbertModel, GPT2Model
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from multiprocessing import freeze_support
from datasets import Dataset
import time
import os
# Depending on your approach, you might need to adapt the structure of this template or parts not marked by TODOs.
# It is not necessary to completely follow this template. Feel free to add more code and delete any parts that 
# are not required 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64  # TODO: Set the batch size according to both training performance and available memory
NUM_EPOCHS = 10  # TODO: Set the number of epochs
LR = 0.01
Gamma = 0.9
TRUNCATION = False
PADDING = True
PREPROCESSING = True

EMBEDDINGS = False

train_val = pd.read_csv("project_4/train.csv")
test_val = pd.read_csv("project_4/test_no_score.csv")


def generate_embeddings(data, train):

    #Distil bert
    #tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    #model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    #Albert
    #tokenizer = AutoTokenizer.from_pretrained("albert/albert-base-v2")
    #model = AlbertModel.from_pretrained("albert/albert-base-v2")

    #gpt-2
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2Model.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    model.to(DEVICE)
    embeddings = []
    scores = []

    dataset = Dataset.from_pandas(data)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)


    start = time.time()
    for i, batch in enumerate(dataloader):
        
        encoded_input = tokenizer(batch["sentence"], return_tensors='pt', padding=PADDING, truncation=TRUNCATION).to(DEVICE)

        with torch.no_grad():
            output = model(**encoded_input)

        output_tensor = output.last_hidden_state
        last_tensor = output_tensor[:, -1, :].cpu()

        if(PREPROCESSING):
            std = torch.std(last_tensor, dim=0)
            mean = torch.mean(last_tensor, dim=0) 
            last_tensor = (last_tensor - mean)/std

        embeddings.append(last_tensor)


        if(train == True):
            scores.append(batch["score"])

        if(i % 5 == 0):
            print(i)
            
    end = time.time()    
    print('Time consumption {} sec'.format(end - start)) 



    embeddings = torch.cat(embeddings, dim=0)
    print(np.shape(embeddings))
    
    """
    if(PREPROCESSING):
        std = torch.std(embeddings, dim=0)
        mean = torch.mean(embeddings, dim=0) 
        embeddings = (embeddings - mean)/std
    """

    if(train == True):
        
        scores = torch.cat(scores, dim=0)
        np.save('project_4/dataset/scores.npy', scores)
        np.save('project_4/dataset/embeddings_train.npy', embeddings)

    else:
        np.save('project_4/dataset/embeddings_test.npy', embeddings)



def get_embeddings(train):
    if(train):
        scores = torch.from_numpy(np.load('project_4/dataset/scores.npy'))
        dataset = torch.from_numpy(np.load('project_4/dataset/embeddings_train.npy'))
        return TensorDataset(dataset, scores)

    else:
        dataset = torch.from_numpy(np.load('project_4/dataset/embeddings_test.npy'))
        return TensorDataset(dataset)



#if we have to create new embeddings
if EMBEDDINGS:
    generate_embeddings(train_val, train = True)
    generate_embeddings(test_val, train = False)
    exit(0)
    train_dataset = get_embeddings(train = True)
    test_dataset = get_embeddings(train = False)
    
else:
    train_dataset = get_embeddings(train = True)
    test_dataset = get_embeddings(train = False)


train_loader = DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=16, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=16, pin_memory=True)

# Additional code if needed

# TODO: Fill out MyModule
class MyModule(nn.Module):
    def __init__(self, dropout=True):
        super().__init__()

        self.fc1 = nn.Sequential(nn.Linear(768, 500), nn.BatchNorm1d(500), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(500, 300), nn.BatchNorm1d(300), nn.ReLU())

        if dropout:
            self.fc3 = nn.Sequential(nn.Dropout(),nn.Linear(300, 1))
        else:
            self.fc3 = nn.Linear(300, 1)

    def forward(self, x):
        x = x.view(-1, 768)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = F.relu(x)
        return x
    
def main():
    
    model = MyModule()
    model.to(DEVICE)
    model.train()
    print(DEVICE)

    # TODO: Setup loss function, optimiser, and scheduler
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ExponentialLR(optimizer, gamma=Gamma)


    training_loss = []
    
    for epoch in range(NUM_EPOCHS): 
        loss_sum = 0
        number_of_batches = 0
        for X_batch, y_batch in tqdm(train_loader, total=len(train_loader)):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            #print("Model Device:", next(model.parameters()).device)
            #print("Input Data Device:", X_batch.device)

            y_pred = model.forward(X_batch)
            loss = criterion(torch.squeeze(y_pred),y_batch.float())
            loss_sum += loss.item()
            number_of_batches += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        training_loss.append(loss_sum/number_of_batches)
        print(f'epoch: {epoch:2}  training_loss: {training_loss[-1]:10.8f}')

    
    model.eval()
    with torch.no_grad():
        results = []
        for [batch] in tqdm(test_loader, total=len(test_loader)):

            # TODO: Set up evaluation loop
            predicted = model(batch.to(DEVICE))
            results.append(predicted.cpu().numpy()) 

        with open("project_4/result.txt", "w") as f:
            for val in np.concatenate(results):
                f.write(f"{val[0]}\n")

if __name__ == '__main__':
    freeze_support()
    main()

