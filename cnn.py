import torch
import torch.nn as nn
import random

class cnn(nn.Module):
    def __init__(self, nodes):
        super().__init__()
        self.conv1 = nn.Conv2d(nodes['conv1'][0],nodes['conv1'][1],nodes['conv1'][2])
        self.pool1 = nn.MaxPool2d(nodes['pool1'])
        self.conv2 = nn.Conv2d(nodes['conv2'][0],nodes['conv2'][1],nodes['conv2'][2])
        self.pool2 = nn.MaxPool2d(nodes['pool2'])
        self.w1 = nn.Linear(nodes['layout'][0], nodes['layout'][1])
        self.w2 = nn.Linear(nodes['layout'][1], nodes['layout'][2])
        self.w3 = nn.Linear(nodes['layout'][2], nodes['layout'][3])

    def forward(self, nodes, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        # return x.shape
        x = x.view(-1, nodes['layout'][0])
        x = torch.relu(self.w1(x))
        x = torch.relu(self.w2(x))
        x = self.w3(x)
        return x
    
    def create(nodes, batch_size, num_epochs, learning_rate, inputs, outputs):
        trainingX0 = []
        testingX0 = []
        trainingY = []
        testingY = []
        tempX = []
        tempY = []

        for i in range(len(inputs)):
            if(i % 5 == 0):
                testingX0.append(inputs[i])
                testingY.append(outputs[i])
            else:
                #append a datapoint or batch of datapoints within an array
                tempX.append(inputs[i])
                tempY.append(outputs[i])
            if(len(tempX) == batch_size):
                trainingX0.append(tempX)
                trainingY.append(tempY)
                tempX = []
                tempY = []
        if(len(tempX) > 0):
            for i in range(len(tempX)):
                testingX0.append(tempX[i])
                testingY.append(tempY[i])
        
        #get number dimensions
        nodes['conv1'] = [len(trainingX0[0][0])] + nodes['conv1']
        #set second dimension from user assigned first
        nodes['conv2'] = [nodes['conv1'][1]] + nodes['conv2']

        #get row dimension
        dim1 = (len(trainingX0[0][0][0]) - nodes['conv1'][2] + 1) / nodes['pool1']
        dim1 = (dim1 - nodes['conv2'][2] + 1) / nodes['pool2']
        #get column dimension
        dim2 = (len(trainingX0[0][0][0][0]) - nodes['conv1'][2] + 1) / nodes['pool1']
        dim2 = (dim2 - nodes['conv2'][2] + 1) / nodes['pool2']


        #set flat layer nodes
        nodes['layout'] = [int(nodes['conv2'][1]*dim1*dim2)] + nodes['layout']
        #set output nodes
        nodes['layout'].append(max(outputs) + 1)


        testingX0 = torch.FloatTensor(testingX0)
        testingY = torch.tensor(testingY)
        trainingX0 = torch.FloatTensor(trainingX0)
        trainingY = torch.tensor(trainingY)
        returnString = ''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = cnn(nodes).to(device)

        criterion = nn.CrossEntropyLoss()
       
        
        for epoch in range(num_epochs):

            for i in range(len(trainingX0)):
                x0 = trainingX0[i].to(device)
                y = trainingY[i].to(device)
                
                outputs = model.forward(nodes, x0)
                
                loss = criterion(outputs, y)

                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if((epoch + 1) % (num_epochs) == 0):
                with torch.no_grad():
                    total = len(testingX0)
                    totalCorrect = 0
                    x0 = testingX0.to(device)
                    y = testingY.to(device)
                    
                    outputs = model.forward(nodes, x0)
                    for i in range(len(outputs)):
                        if(torch.argmax(outputs[i]) == y[i]):
                            totalCorrect = totalCorrect + 1
                    returnString = returnString + 'Iteration #' + str(epoch+1) + ':\nTraining Loss: \n' + str(loss.item()) +'\nTesting percent: \n' + str(totalCorrect/total * 100) + '%\n'

        fileNumber = str(random.randint(0, 1000000000))

        returnString = returnString  + '\n' + 'Model id: ' + fileNumber
        
        torch.save(model.state_dict(), './models/' + fileNumber + '.pth')
        return returnString


    def run(modelId, x):
        tempModel = torch.load('./models/' + str(modelId) + '.pth')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = cnn().to(device)
        model.load_state_dict(tempModel)
        model.eval()

        with torch.no_grad():
            X0 = torch.FloatTensor(x)
            outputs = model.forward(X0)
        return torch.argmax(outputs, dim=1).tolist()

    def improve(modelId, batch_size, num_epochs, learning_rate, inputs, outputs):
        trainingX0 = []
        testingX0 = []
        trainingY = []
        testingY = []
        tempX = []
        tempY = []

        for i in range(len(inputs)):
            if(i % 5 == 0):
                testingX0.append(inputs[i])
                testingY.append(outputs[i])
            else:
                #append a datapoint or batch of datapoints within an array
                tempX.append(inputs[i])
                tempY.append(outputs[i])
            if(len(tempX) == batch_size):
                trainingX0.append(tempX)
                trainingY.append(tempY)
                tempX = []
                tempY = []
        if(len(tempX) > 0):
            for i in range(len(tempX)):
                testingX0.append(tempX[i])
                testingY.append(tempY[i])

        testingX0 = torch.FloatTensor(testingX0)
        testingY = torch.tensor(testingY)
        trainingX0 = torch.FloatTensor(trainingX0)
        trainingY = torch.tensor(trainingY)
        returnString = ''
        tempModel = torch.load('./models/' + str(modelId) + '.pth')

    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = cnn().to(device)
        model.load_state_dict(tempModel)
        model.eval()

        criterion = nn.CrossEntropyLoss()
       
        
        for epoch in range(num_epochs):
            for i in range(len(trainingX0)):
                x0 = trainingX0[i].to(device)
                y = trainingY[i].to(device)
                
                outputs = model.forward(x0)
                
                loss = criterion(outputs, y)

                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if((epoch + 1) % (num_epochs/10) == 0):
                with torch.no_grad():
                    total = len(testingX0)
                    totalCorrect = 0
                    x0 = testingX0.to(device)
                    y = testingY.to(device)
                    
                    outputs = model.forward(x0)
                    for i in range(len(outputs)):
                        if(torch.argmax(outputs[i]) == y[i]):
                            totalCorrect = totalCorrect + 1
                    returnString = returnString + 'Iteration #' + str(epoch+1) + ':\nTraining Loss: \n' + str(loss.item()) +'\nTesting percent: \n' + str(totalCorrect/total * 100) + '%\n'

        fileNumber = str(modelId)

        returnString = returnString  + '\n' + 'Model id: ' + fileNumber
        
        torch.save(model.state_dict(), './models/' + fileNumber + '.pth')
        return returnString