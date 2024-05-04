
import numpy as np
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

parameters = "lr = 0.0006, epochs = 25, accuracy = "


with open('project_3/solution_p.txt', 'r') as file:

    text = file.read()

text = text.replace("tensor([", "").replace(".])", "")

sample = np.fromstring(text, dtype=np.int32, sep="\n")
print(len(sample))

with open('project_3/results_p.txt', 'r') as file:

    text = file.read()

pred = np.fromstring(text, dtype=np.int32, sep="\n")
print("predictions")
print(len(pred))

mistakes = 0
for i in range(0,len(pred)):
    if(pred[i] != sample[i]):
        
        mistakes += 1

accuracy = (len(sample)-mistakes)/len(sample)

print("mistakes: %s" %{mistakes})
print("number of samples: %s" %{len(sample)})
print("accuracy: %s" %{accuracy})

with open('project_3/results_list.txt', 'r+') as file:
    text = file.read()
    #file.write(parameters + str(accuracy) + "\n")
    #print(text + parameters + str(accuracy) + "\n")


