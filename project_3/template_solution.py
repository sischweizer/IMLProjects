# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, resnet152, ResNet152_Weights
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import torch
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import time
from sklearn.model_selection import train_test_split
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")
# When using the GPU, it is important that your model and all data are on the 
# same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size=64

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # TODO: define a transform to pre-process the images
    # The required pre-processing depends on the pre-trained model you choose 
    # below. 
    # See https://pytorch.org/vision/stable/models.html#using-the-pre-trained-models
    weights = ResNet152_Weights.DEFAULT
    train_transforms = transforms.Compose([transforms.ToTensor(), weights.transforms()])
    train_dataset = datasets.ImageFolder(root="project_3/dataset/", transform=train_transforms)
    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't 
    # run out of memory (VRAM if on GPU, RAM if on CPU)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=64,
                              shuffle=False,
                              pin_memory=True, 
                              num_workers=16)
    
    # TODO: define a model for extraction of the embeddings (Hint: load a pretrained model,
    #  more info here: https://pytorch.org/vision/stable/models.html)
    #model = nn.Module()

    #model = resnet50(weights=ResNet50_Weights.DEFAULT, progress=True)
    model = resnet152(weights=weights, progress=True)
    
    #model = resnet101(weights=ResNet101_Weights.DEFAULT, progress=True)
    model.to(device)
    embedding_size = 2048 # Dummy variable, replace with the actual embedding size once you 
    # pick your model
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))
    # TODO: Use the model to extract the embeddings. Hint: remove the last layers of the 
    # model to access the embeddings the model generates. 
    
    #train_nodes, eval_nodes = get_graph_node_names(resnet50())
    #print(train_nodes)
    model_2 = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)
    
    #return_nodes = {
    #    'fc': 'layer4',
    #}

    #model_2 = create_feature_extractor(model, return_nodes=return_nodes)
    #model_2.to(device)

    #for i in range(0,len(train_dataset.imgs)):
    #    model_2(train_dataset.imgs[i])
    start = time.time()
    with torch.no_grad():
        for i, img in enumerate(train_loader.dataset):
            print(i)
            
            
            data = (img[0]).to(device)
            result = model_2(data)
            embeddings[i] = torch.flatten(result.cpu())
    end = time.time()    
    print('Time consumption {} sec'.format(end - start))    
     
    
    
    np.save('project_3/dataset/embeddings.npy', embeddings)
    

def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="project_3/dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].split('\\')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('project_3/dataset/embeddings.npy')
    # TODO: Normalize the embeddings
    
    
    embeddings = torch.from_numpy(embeddings)
    std = torch.std(embeddings, dim=0)
    mean = torch.mean(embeddings, dim=0) 
    embeddings = (embeddings - mean)/std

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# Hint: adjust batch_size and num_workers to your PC configuration, so that you 
# don't run out of memory (VRAM if on GPU, RAM if on CPU)
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        # Attention: If you get type errors you can modify the type of the
        # labels here
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader

# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self, dropout=True):
        """
        The constructor of the model.
        """
        super().__init__()

        self.fc1 = nn.Sequential(nn.Linear(6144, 1000), nn.BatchNorm1d(1000), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(1000, 400), nn.BatchNorm1d(400), nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(400, 200), nn.BatchNorm1d(200), nn.ReLU())
        #self.fc4 = nn.Sequential(nn.Linear(800, 400), nn.BatchNorm1d(400), nn.LeakyReLU())

        if dropout:
            self.fc5 = nn.Sequential(nn.Dropout(),nn.Linear(200, 1))
        else:
            self.fc5 = nn.Linear(200, 1)
        #torch.nn.init.kaiming_normal_(self.fc5.weight, mode='fan_out', nonlinearity='relu')
        

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = x.view(-1, 6144)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        #x = self.fc4(x)
        x = self.fc5(x)
        return x

def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    n_epochs = 10
    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part 
    # of the training data as a validation split. After each epoch, compute the loss on the 
    # validation split and print it out. This enables you to see how your model is performing 
    # on the validation data before submitting the results on the server. After choosing the 
    # best model, train it on the whole training data.


    #validation set split
    g = torch.Generator(device="cpu")
    data_train, data_test = random_split(train_loader.dataset, [0.8, 0.2], generator= g)

    training_set = DataLoader(list(data_train), shuffle = False, batch_size=batch_size)
    validation_set = DataLoader(list(data_test), shuffle = False, batch_size=batch_size)


    loss_fct = torch.nn.BCEWithLogitsLoss()
    training_loss = []
    validation_loss = []

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    #scheduler = StepLR(optimizer,step_size=100, gamma=0.9)
    #scheduler = ExponentialLR(optimizer, gamma=0.9)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.006, momentum=0.9)
    start = time.time()

    tot_size = 6144
    
    #training 
    for epoch in range(n_epochs): 
        loss_sum = 0
        number_of_batches = 0
        for X_batch, y_batch in (training_set):
            y_pred = model.forward(X_batch.to(device))
            #print(y_batch.size())
            #print(torch.squeeze(y_pred).size)
            loss = loss_fct(torch.squeeze(y_pred),y_batch.float().to(device))
            loss_sum += loss.item()
            number_of_batches += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

        training_loss.append(loss_sum/number_of_batches)
        print(f'epoch: {epoch:2}  training_loss: {training_loss[-1]:10.8f}')
        number_of_batches = 0
        loss_sum = 0
        for X_batch, y_batch in (validation_set):
            y_pred = model.forward(X_batch.to(device))
            loss = loss_fct(torch.squeeze(y_pred),y_batch.float().to(device))
            loss_sum += loss.item()
            number_of_batches += 1

        validation_loss.append(loss_sum/number_of_batches)

        print(f'epoch: {epoch:2}  validation_loss: {validation_loss[-1]:10.8f}')
            ##############

        #x_train = torch.empty(len(training_set),tot_size)
        #y_train = torch.empty(len(training_set),1)
        #for i, [X, y] in enumerate(training_set):
        #    x_train[i] = X
        #    y_train[i] = y

        #y_pred = model.forward(x_train.to(device))

        #loss = loss_fct(y_pred,y_train.float().to(device))
        #training_loss.append(loss)

        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #scheduler.step()
        #print(f'epoch: {epoch:2}  training_loss: {loss.item():10.8f}')
        
        #x_val = torch.empty(len(validation_set),tot_size)
        #y_val = torch.empty(len(validation_set),1)
        #for i, [X, y] in enumerate(validation_set):
        #    x_val[i] = X
        #    y_val[i] = y
        #y_pred = model.forward(x_val.to(device))     
        #loss = loss_fct(y_pred,y_val.float().to(device)) 
        #validation_loss.append(loss)
        #print(f'epoch: {epoch:2}  validation_loss: {loss.item():10.8f}')
        
        end = time.time()    
        print('Time consumption {} sec'.format(end - start)) 
        start = time.time()

          
    
    #print("training loss:")
    #print(training_loss)
    #print("validation loss:")
    #print(validation_loss)
    
    loss_tot = []
    for epoch in range(n_epochs):  
        number_of_batches = 0      
        loss_sum = 0
        for X_batch, y_batch in (train_loader):
            y_pred = model.forward(X_batch.to(device))
            loss = loss_fct(torch.squeeze(y_pred),y_batch.float().to(device))
            loss_sum += loss.item()
            number_of_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()

        loss_tot.append(loss_sum/number_of_batches)
        print(f'epoch: {epoch:2}  total_loss: {loss_tot[-1]:10.8f}')

        """x_tot = torch.empty(len(train_loader.dataset),tot_size)
        y_tot = torch.empty(len(train_loader.dataset),1)
        for i, [X, y] in enumerate(train_loader.dataset):
            x_tot[i] = X
            y_tot[i] = y
        y_pred = model.forward(x_tot.to(device))
             
        loss = loss_fct(y_pred,y_tot.float().to(device)) 
        loss_tot.append(loss)
        print(f'epoch: {epoch:2}  total_loss: {loss.item():10.8f}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        """
    #print(loss_tot)
    return model

def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """

    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
            
        predictions = np.vstack(predictions)
 
    np.savetxt("project_3/results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'project_3/train_triplets.txt'
    TEST_TRIPLETS = 'project_3/test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('project_3/dataset/embeddings.npy') == False):
        generate_embeddings()
        print("finished embedingspart")

    # load the training data
    X, y = get_data(TRAIN_TRIPLETS)
    
    
    # Create data loaders for the training data
    train_loader = create_loader_from_np(X, y, train = True, batch_size=64)
    # delete the loaded training data to save memory, as the data loader copies
    del X
    del y

    # repeat for testing data
    X_test, y_test = get_data(TEST_TRIPLETS, train=False)

    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)
    del X_test
    del y_test

    # define a model and train it
    model = train_model(train_loader)
    
    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")
