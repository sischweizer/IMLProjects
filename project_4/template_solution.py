# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:

from multiprocessing import freeze_support
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from datasets import Dataset as Ds
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim.lr_scheduler import ExponentialLR

# Depending on your approach, you might need to adapt the structure of this template or parts not marked by TODOs.
# It is not necessary to completely follow this template. Feel free to add more code and delete any parts that 
# are not required 

EMBEDDING_TOKENIZER = 'distilbert/distilbert-base-uncased'
EMBEDDING_MODEL = "distilbert/distilbert-base-uncased"

EMBEDDING_SIZE = 768

GENERATE_EMBEDDINGS = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(DEVICE)
NUM_WORKERS = 8
BATCH_SIZE = 64  # TODO: Set the batch size according to both training performance and available memory
NUM_EPOCHS = 1  

LR = 0.001
GAMMA = 0.9

train_val = pd.read_csv("train.csv")
test_val = pd.read_csv("test_no_score.csv")

# TODO: Fill out the ReviewDataset
class ReviewDataset(Dataset):
    def __init__(self, data_frame):
        if(len(data_frame.columns) == 3):
            print("generating train ReviewDataset")
            self.train = True
            self.path = 'dataset/train_embeddings.npy'
        elif(len(data_frame.columns) == 2):
            print("generating test ReviewDataset")
            self.train = False
            self.path = 'dataset/test_embeddings.npy'
        else:
            print("invalid data frame size")
            exit(1)

        '''
        if (self.train):
            self.path = 'dataset/train_embeddings.npy'
        else:
            self.path = 'dataset/test_embeddings.npy'
        '''

        if (os.path.exists(self.path) == False or GENERATE_EMBEDDINGS):
            self.dataset = self.generate_embeddings(data_frame)
        else:
            self.dataset = torch.from_numpy(np.load(self.path))

        #self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if (self.train):
            x = self.dataset[index][0:EMBEDDING_SIZE]
            y = self.dataset[index][-1]
            return (x, y)
        else:
            return self.dataset[index]
        

    def generate_embeddings(self, data_frame):
        
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_TOKENIZER)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL, torch_dtype=torch.float16)
        #model = AutoModel.from_pretrained(EMBEDDING_MODEL, torch_dtype=torch.float16, attn_implementation="flash_attention_2)"
        model.to(DEVICE)
        embeddings = []

        #DEBUG
        #print(type(data_frame['sentence']))

        dataset = Ds.from_pandas(data_frame)['sentence']

        print("generating embeddings")
        data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)

        for i, batch in enumerate(data_loader):
            print(i)
            encoded_input = tokenizer(batch, return_tensors='pt', padding=True).to(DEVICE)

            with torch.no_grad():
                output = model(**encoded_input)

            output_tensor = output.last_hidden_state
            #last_tensor = output_tensor[:][-1][:].cpu()
            last_tensor = torch.select(output_tensor, 1, -1).cpu()
            embeddings.append(last_tensor)

            #DEBUG
            #if i == 10:
            #    break

        embeddings_tensor = torch.cat(embeddings)

        #normalize smbeddings
        std = torch.std(embeddings_tensor, dim=0)
        mean = torch.mean(embeddings_tensor, dim=0) 
        embeddings_tensor = (embeddings_tensor - mean)/std

        #print(np.shape(embeddings_tensor))
        
        if (self.train):
            #scores = torch.tensor(Ds.from_pandas(data_frame)['score'][0:(11*BATCH_SIZE)]).unsqueeze(1)
            scores = torch.tensor(Ds.from_pandas(data_frame)['score']).unsqueeze(1)
            #result = list(zip(embeddings_tensor, Ds.from_pandas(data_frame)['score'][0:(11*BATCH_SIZE)]))
            result = torch.cat((embeddings_tensor, scores), dim=1)
            #print(np.shape(result))
        else:
            result = embeddings_tensor

        np.save(self.path, result)
        return result
        

    

    #def load_embeddings(self, path):
    #    self.dataset = torch.from_numpy(np.load(path))


train_dataset = ReviewDataset(train_val)
test_dataset = ReviewDataset(test_val)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Additional code if needed

# TODO: Fill out MyModule
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(EMBEDDING_SIZE, 500), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(500, 250), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(250, 250), nn.ReLU())
        self.fc4 = nn.Sequential(nn.Dropout(),nn.Linear(250, 1), nn.ReLU())

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


model = MyModule().to(DEVICE)

# TODO: Setup loss function, optimiser, and scheduler
criterion = nn.MSELoss()
optimiser = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ExponentialLR(optimiser, gamma=GAMMA)

def main():
    print("training model")

    model.train()
    for epoch in range(NUM_EPOCHS):
        print(f'epoch: {epoch}')
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, total=len(train_loader)):
            #batch = batch.to(DEVICE)
            [x_batch, y_batch] = batch
            y_pred = model.forward(x_batch.to(DEVICE))

            loss = criterion(y_pred, y_batch.to(DEVICE))

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            train_loss += loss.item() * len(x_batch)
            # TODO: Set up training loop

        scheduler.step()

        train_loss = train_loss / len(train_loader.dataset)
        print(f'training loss: {train_loss}')

    print("generating predictions")

    model.eval()
    with torch.no_grad():
        results = []
        for batch in tqdm(test_loader, total=len(test_loader)):
            #[x_batch] = batch
            prediction = model(batch.float().to(DEVICE))
            results.append(prediction.squeeze().cpu())

            # TODO: Set up evaluation loop

        #results.cpu()
        with open("result.txt", "w") as f:
            for val in np.concatenate(results):
                f.write(f"{val}\n")


if __name__ == '__main__':
    freeze_support()
    main()