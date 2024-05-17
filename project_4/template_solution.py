# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from datasets import Dataset as Ds
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# Depending on your approach, you might need to adapt the structure of this template or parts not marked by TODOs.
# It is not necessary to completely follow this template. Feel free to add more code and delete any parts that 
# are not required 

EMBEDDING_TOKENIZER = 'distilbert/distilbert-base-uncased'
EMBEDDING_MODEL = "distilbert/distilbert-base-uncased"

EMBEDDING_SIZE = 768

GENERATE_EMBEDDINGS = False

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
NUM_WORKERS = 8
BATCH_SIZE = 64  # TODO: Set the batch size according to both training performance and available memory
NUM_EPOCHS = None  # TODO: Set the number of epochs

train_val = pd.read_csv("train.csv")
test_val = pd.read_csv("test_no_score.csv")

# TODO: Fill out the ReviewDataset
class ReviewDataset(Dataset):
    def __init__(self, data_frame, train):
        if (train):
            path = 'dataset/train_embeddings.npy'
        else:
            path = 'dataset/test_embeddings.npy'

        if (os.path.exists(path) == False or GENERATE_EMBEDDINGS):
            self.dataset = self.generate_embeddings(data_frame, train)
        else:
            self.dataset = torch.from_numpy(np.load(path))

        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if (self.train):
            x = self.dataset[index][0:EMBEDDING_SIZE]
            y = self.dataset[index][-1]
            return (x, y)
        else:
            return self.dataset[index]
        

    def generate_embeddings(self, data_frame, train):
        
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
            if i == 10:
                break

        embeddings_tensor = torch.cat(embeddings)

        #normalize smbeddings
        std = torch.std(embeddings_tensor, dim=0)
        mean = torch.mean(embeddings_tensor, dim=0) 
        embeddings_tensor = (embeddings_tensor - mean)/std

        print(np.shape(embeddings_tensor))
        
        if (train):
            scores = torch.tensor(Ds.from_pandas(data_frame)['score'][0:(11*BATCH_SIZE)]).unsqueeze(1)
            #result = list(zip(embeddings_tensor, Ds.from_pandas(data_frame)['score'][0:(11*BATCH_SIZE)]))
            result = torch.cat((embeddings_tensor, scores), dim=1)
            #print(np.shape(result))
        else:
            result = embeddings_tensor

        return result
        

    

    #def load_embeddings(self, path):
    #    self.dataset = torch.from_numpy(np.load(path))


train_dataset = ReviewDataset(train_val, True)
test_dataset = ReviewDataset(test_val, False)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=16, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=16, pin_memory=True)

# Additional code if needed

# TODO: Fill out MyModule
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


model = MyModule().to(DEVICE)

# TODO: Setup loss function, optimiser, and scheduler
criterion = None
optimiser = None
scheduler = None

model.train()
for epoch in range(NUM_EPOCHS):
    model.train()
    for batch in tqdm(train_loader, total=len(train_loader)):
        batch = batch.to(DEVICE)

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
