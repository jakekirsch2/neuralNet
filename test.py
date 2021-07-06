import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cnn 
import requests
import json

inputs = []
outputs = []




# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 1
batch_size = 4
learning_rate = 0.001

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                          shuffle=True)



# nodes = {
#     'conv1': [6,5],
#     'pool1': 2,
#     'conv2': [16,5],
#     'pool2': 2,
#     'layout': [120, 84]
# }

# inputs = []
# outputs = []
# url = 'http://127.0.0.1:5000/fully-connected/create'
# data = json.dumps({
#     'layout':[3072, 500, 200, 100, 10]
# })
# r = requests.post(url = url, data = data)

modelId = '46103508'

for i, (images, labels) in enumerate(train_dataset):
    inputs.append(images.tolist())
    outputs.append(labels)
    if(i != 0 and i % 100 == 0):
        url = 'http://127.0.0.1:5000/fully-connected/train'

        data = json.dumps({
            'model_id':modelId,
            'batch_size':100,
            'epochs':10,
            'learning_rate':0.01,
            'inputs': inputs,
            'outputs': outputs
        })

        print('sending to server...')

        r = requests.post(url = url, data = data)

        print(str(r.text))
        inputs = []
        outputs = []

# for i, (images, labels) in enumerate(train_dataset):
#     inputs.append(images.tolist())
#     outputs.append(labels)
#     if(i != 0 and i % 10 == 0):
#         url = 'http://127.0.0.1:5000/fully-connected/test'

#         data = json.dumps({
#             'model_id':modelId,
#             'inputs': inputs,
#             'outputs': outputs
#         })

#         print('sending to server...')

#         r = requests.post(url = url, data = data)

#         print(str(r.text))
#         inputs = []
#         outputs = []

# for i, (images, labels) in enumerate(train_dataset):
#     inputs.append(images.tolist())
#     outputs.append(labels)
#     if(i != 0 and i % 10 == 0):
#         url = 'http://127.0.0.1:5000/fully-connected/run'

#         data = json.dumps({
#             'model_id':modelId,
#             'inputs': inputs
#         })

#         print('sending to server...')

#         r = requests.post(url = url, data = data)

#         print(str(r.text))
#         inputs = []
#         outputs = []





# print(cnn.cnn.create(nodes,batch_size, num_epochs, learning_rate, inputs, outputs))
