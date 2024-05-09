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
# Depending on your approach, you might need to adapt the structure of this template or parts not marked by TODOs.
# It is not necessary to completely follow this template. Feel free to add more code and delete any parts that 
# are not required 

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64  # TODO: Set the batch size according to both training performance and available memory
NUM_EPOCHS = 10  # TODO: Set the number of epochs
LR = 0.01
Gamma = 0.9

embeddings = True

train_val = pd.read_csv("project_4/train.csv")
test_val = pd.read_csv("project_4/test_no_score.csv")

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


    if(train == True):
        scores = torch.tensor(scores)
    
        dataset = TensorDataset(result, scores)
        return dataset
    
    else: 
        return TensorDataset(result)


def get_embeddings():
    return "hello"

# TODO: Fill out the ReviewDataset
class ReviewDataset(Dataset):
    def __init__(self, data_frame, train = False):
        
        #generate embeddings
        if(embeddings == True):
            data = generate_embeddings(data_frame, train)
        else: 
            data = get_embeddings()

        self.data = data

        

    def __len__(self):
        print(len(self.data))
        return len(self.data)


    def __getitem__(self, index):
        input_data, target = self.data[index]
        return input_data, target
        


train_dataset = ReviewDataset(train_val, train = True)

test_dataset = ReviewDataset(test_val)


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
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ExponentialLR(optimiser, gamma=Gamma)

    model.train()


    for epoch in range(NUM_EPOCHS):
        model.train()

        for  batch in tqdm(train_loader, total=len(train_loader)):
            batch = batch.to(DEVICE)
            print("5")
            
            
            # TODO: Set up training loop


    model.eval()
    with torch.no_grad():
        results = []
        for batch in tqdm(test_loader, total=len(test_loader)):
            batch = batch.to(DEVICE)

            # TODO: Set up evaluation loop

        with open("result.txt", "w") as f:
            for val in np.concatenate(results):
                f.write(f"{val}\n")

if __name__ == '__main__':
    freeze_support()
    main()

