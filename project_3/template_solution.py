# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import time

from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

# The device is automatically set to GPU if available, otherwise CPU
# If you want to force the device to CPU, you can change the line to
# device = torch.device("cpu")
# When using the GPU, it is important that your model and all data are on the 
# same device.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # TODO: define a transform to pre-process the images
    # The required pre-processing depends on the pre-trained model you choose 
    # below. 
    # See https://pytorch.org/vision/stable/models.html#using-the-pre-trained-models
    train_transforms = transforms.Compose([transforms.ToTensor()])
    
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
    model = resnet50(weights=ResNet50_Weights.DEFAULT, progress=True)
    model.to(device)
    embedding_size = 1000 # Dummy variable, replace with the actual embedding size once you 
    # pick your model
    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))
    # TODO: Use the model to extract the embeddings. Hint: remove the last layers of the 
    # model to access the embeddings the model generates. 
    
    #train_nodes, eval_nodes = get_graph_node_names(resnet50())
    #print(train_nodes)

    return_nodes = {
        'fc': 'layer4',
    }

    model_2 = create_feature_extractor(model, return_nodes=return_nodes)
    model_2.to(device)

    #for i in range(0,len(train_dataset.imgs)):
    #    model_2(train_dataset.imgs[i])
    start = time.time()
    with torch.no_grad():
        for i, img in enumerate(train_loader.dataset):
            print(i)
            data = (img[0].unsqueeze(0)).to(device)
            result = model_2(data)['layer4']
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
    std = torch.std(embeddings)
    mean = torch.mean(embeddings) 
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

        self.fc1 = nn.Sequential(nn.Linear(3000, 750), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(750, 375), nn.ReLU())
        if dropout:
            self.fc3 = nn.Sequential(nn.Dropout(), nn.Linear(375, 1))
        else:
            self.fc3 = nn.Linear(375, 1)
        

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = x.view(-1, 3000)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.fc3(x)
        x = F.sigmoid(x)
        #x = F.relu(x)
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
    training_set,validation_set = random_split(train_loader.dataset, [0.8, 0.2], generator= g)

    loss_fct = torch.nn.BCEWithLogitsLoss()
    training_loss = []
    validation_loss = []
    print(type(training_set))
    print(type(training_set.dataset))
    print(len(training_set.dataset[0][0]))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    #training --> backward propagation is missing
    for epoch in range(n_epochs):   
        print(epoch)      
        for [X, y] in training_set:
            
            y_pred = model.forward(X.to(device))
            loss = loss_fct(torch.flatten(y_pred)[0].clamp(0, 1),y.float().to(device))
            training_loss.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for [X, y] in validation_set:
            y_pred = model.forward(X.to(device))     
            loss = loss_fct(torch.flatten(y_pred)[0].clamp(0, 1),y.float().to(device)) 
            validation_loss.append(loss)
        print(loss)

          

    print("training loss:")
    print(training_loss)
    print("validation loss:")
    print(validation_loss)

    loss = []
    for epoch in range(n_epochs):        
        for [X, y] in train_loader:
            y_pred = model.forward(X.to(device))
            loss = loss_fct(torch.flatten(y_pred)[0].clamp(0, 1),y.float().to(device))
            training_loss.append(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pass
    print(loss)
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
