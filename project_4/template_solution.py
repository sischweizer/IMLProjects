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

EMBEDDINGS = False

train_val = pd.read_csv("project_4/train.csv")
test_val = pd.read_csv("project_4/test_no_score.csv")
"""
def generate_embeddings(data, train):

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.to(DEVICE)

    
    scores  = []
    for index, row in data.iterrows():
        encoded_input = tokenizer(row["sentence"], return_tensors='pt').to(DEVICE)
        output = model(**encoded_input)
        output_tensor = output.last_hidden_state
        last_tensor = torch.squeeze(output_tensor)[-1].unsqueeze(0)

        if(train == True):
            scores.append(row["score"])

        if(index == 0):
            result = last_tensor
            
        else: 
            result = torch.cat((result, last_tensor),dim=0)
        
        if(index % 10 == 0):
            print(index)


    if(train == True):
        scores = torch.tensor(scores)
    
        dataset = TensorDataset(result, scores)
        return dataset
    
    else: 
        return TensorDataset(result)
"""
def generate_embeddings(data, train):

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    model.to(DEVICE)

    #print(data.items)
    #print(data.columns)
    
    embeddings = []
    scores = []

    dataset = Dataset.from_pandas(data)
    #print(dataset)
    #print(type(dataset))

    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)
    #print(type(dataloader))

    start = time.time()
    for i, batch in enumerate(dataloader):
        #print(i)
        #print(batch)
        #print(type(batch))
        
        encoded_input = tokenizer(batch["sentence"], return_tensors='pt', padding=True).to(DEVICE)

        with torch.no_grad():
            output = model(**encoded_input)

        output_tensor = output.last_hidden_state
        #print(np.shape(output_tensor))
        last_tensor = output_tensor[:, -1, :].cpu()
        #print(np.shape(last_tensor))

        #last_tensor = torch.squeeze(output_tensor)[-1].unsqueeze(0)

        embeddings.append(last_tensor)
        #print(np.shape(last_tensor))

        if(train == True):
            scores.append(batch["score"])

        if(i % 5 == 0):
            print(i)
            
    end = time.time()    
    print('Time consumption {} sec'.format(end - start)) 

    embeddings = torch.cat(embeddings, dim=0)
    print(np.shape(embeddings))
    
    

    if(train == True):
        
        scores = torch.cat(scores, dim=0)
        np.save('project_4/dataset/scores.npy', scores)
        #print(np.shape(scores))
        #scores = torch.tensor(scores)  
        np.save('project_4/dataset/embeddings_train.npy', embeddings)
        dataset = TensorDataset(embeddings, scores)

    else:
        np.save('project_4/dataset/embeddings_test.npy', embeddings)
        dataset = TensorDataset(embeddings)
    

    return dataset


def get_embeddings(train):
    if(train):
        scores = torch.from_numpy(np.load('project_4/dataset/scores.npy'))
        dataset = torch.from_numpy(np.load('project_4/dataset/embeddings_train.npy'))
        return TensorDataset(dataset, scores)

    else:
        dataset = torch.from_numpy(np.load('project_4/dataset/embeddings_test.npy'))
        return TensorDataset(dataset)

"""
# TODO: Fill out the ReviewDataset
class ReviewDataset(Dataset):
    def __init__(self, data_frame, train=False):
        if EMBEDDINGS:
            self.dataset = generate_embeddings(data_frame, train)
        else:
            self.dataset = get_embeddings(train)
        self.train = train

    def __getitem__(self, index):
        
        start = (index[0] -1) * BATCH_SIZE
        end = start + BATCH_SIZE

        if self.train:
            
            input_data = self.dataset[start:end][0]

            label = self.dataset[start:end][1]
            #print(label)
            #print(input_data)
            return input_data, label
        else:
            input_data = self.dataset[index][0]
            return input_data, torch.tensor(-1)  # Dummy label for test set

    def __len__(self):
        return len(self.dataset)

"""

#if we have to create new embeddings
if EMBEDDINGS:
    train_dataset = generate_embeddings(train_val, train = True)
    test_dataset = generate_embeddings(train_val, train = False)
else:
    train_dataset = get_embeddings(train = True)
    test_dataset = get_embeddings(train = False)



#train_dataset = ReviewDataset(train_val, train = True)

#test_dataset = ReviewDataset(test_val)


train_loader = DataLoader(dataset=train_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=16, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=16, pin_memory=True)

#print(train_dataset.__getitem__(16))


# Additional code if needed

# TODO: Fill out MyModule
class MyModule(nn.Module):
    def __init__(self, dropout=True):
        super().__init__()

        self.fc1 = nn.Sequential(nn.Linear(768, 400), nn.BatchNorm1d(400), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(400, 200), nn.BatchNorm1d(200), nn.ReLU())

        if dropout:
            self.fc3 = nn.Sequential(nn.Dropout(),nn.Linear(200, 1))
        else:
            self.fc3 = nn.Linear(200, 1)

    def forward(self, x):
        x = x.view(-1, 768)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
def main():
    
    model = MyModule().to(DEVICE)

    # TODO: Setup loss function, optimiser, and scheduler
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ExponentialLR(optimizer, gamma=Gamma)



    model.train()
    
    """
    for epoch in range(NUM_EPOCHS):
        model.train()
        
        for batch in tqdm(train_loader, total=len(train_loader)):
            batch = batch.to(DEVICE)
            print("5")
            
            
            # TODO: Set up training loop
        """
    training_loss = []
    
    for epoch in range(NUM_EPOCHS): 
        loss_sum = 0
        number_of_batches = 0
        for X_batch, y_batch in tqdm(train_loader, total=len(train_loader)):

            
            y_pred = model.forward(X_batch.to(DEVICE))
            loss = criterion(torch.squeeze(y_pred),y_batch.float().to(DEVICE))
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

