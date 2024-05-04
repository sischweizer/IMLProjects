
import numpy as np

parameters = "lr = 0.0006, epochs = 25, accuracy = "


with open('project_3/sample.txt', 'r') as file:

    text = file.read()

sample = np.fromstring(text, dtype=np.int32, sep="\n")
print(len(sample))

with open('project_3/results.txt', 'r') as file:

    text = file.read()

pred = np.fromstring(text, dtype=np.int32, sep="\n")
print(len(pred))

mistakes = 0
for i in range(0,len(pred)):
    if(pred[i] != sample[i]):
        print("mistake")
        print(pred[i])
        print(sample[i])
        mistakes += 1

accuracy = (len(sample)-mistakes)/len(sample)

print("mistakes: %s" %{mistakes})
print("number of samples: %s" %{len(sample)})
print("accuracy: %s" %{accuracy})

with open('project_3/results_list.txt', 'r+') as file:
    text = file.read()
    file.write(parameters + str(accuracy) + "\n")
    #print(text + parameters + str(accuracy) + "\n")
