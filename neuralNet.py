import torch
import torch.nn as nn
import random

class neuralNet(nn.Module):
    def __init__(self, nodes):
        super().__init__()

        self.drop = torch.nn.Dropout(0.5)
        
        for i in range(len(nodes) - 1):
            name = 'w' + str(i)
            setattr(self, name, nn.Linear(nodes[i], nodes[i + 1]))

    def forward(self, nodes, x, train=False):
        x = x.view(-1, nodes[0])
        for i in range(len(nodes) - 1):
            name = 'w' + str(i)
            if(train):
                x = self.drop(x)
            x = torch.relu(getattr(self, name)(x))
        return x

    def create(nodes):
        
        model = neuralNet(nodes)

        fileNumber = str(random.randint(0, 1000000000))
        
        torch.save(model.state_dict(), './models/' + fileNumber + '.pth')
        return fileNumber

    def run(modelId, x):
        tempModel = torch.load('./models/' + modelId + '.pth')
        nodes = neuralNet.getNodes(tempModel)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = neuralNet(nodes).to(device)
        model.load_state_dict(tempModel)
        model.eval()

        with torch.no_grad():
            X0 = torch.FloatTensor(x)
            outputs = model.forward(nodes, X0)
        return str(torch.argmax(outputs, dim=1).tolist())

    def trainModel(modelId, batch_size, num_epochs, learning_rate, inputs, outputs):
        trainingX0 = []
        trainingY = []
        tempX = []
        tempY = []

        for i in range(len(inputs)):
            tempX.append(inputs[i])
            tempY.append(outputs[i])
            if(len(tempX) == batch_size):
                trainingX0.append(tempX)
                trainingY.append(tempY)
                tempX = []
                tempY = []
        
        trainingX0 = torch.FloatTensor(trainingX0)
        trainingY = torch.tensor(trainingY)
        returnJson = {}

        tempModel = torch.load('./models/' + modelId + '.pth')
        nodes = neuralNet.getNodes(tempModel)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = neuralNet(nodes).to(device)
        model.load_state_dict(tempModel)
        
        criterion = nn.CrossEntropyLoss()
       
        
        for epoch in range(num_epochs):
            for i in range(len(trainingX0)):
    
                x0 = trainingX0[i].to(device)
                y = trainingY[i].to(device)
                
                outputs = model.forward(nodes, x0, True)
                
                loss = criterion(outputs, y)

                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        returnJson['training_loss'] = round(loss.item(),4)

        returnJson['model_id'] = modelId
        
        torch.save(model.state_dict(), './models/' + modelId + '.pth')
        return returnJson
    
    def getNodes(tempModel):
        nodes = []
        nodes.append(len(tempModel['w0.weight'][0]))

        indexIsSet = True
        index = 0

        while indexIsSet:
            tempString = 'w' + str(index) +'.weight'
            try:
                nodes.append(len(tempModel[tempString]))
                index = index + 1
            except:
                indexIsSet = False
        return nodes

    def test(modelId, inputs, outputs):
        testingX0 = torch.FloatTensor(inputs)
        testingY = torch.tensor(outputs)
        returnJson = {}

        tempModel = torch.load('./models/' + modelId + '.pth')
        nodes = neuralNet.getNodes(tempModel)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = neuralNet(nodes).to(device)
        model.load_state_dict(tempModel)
        model.eval()

        with torch.no_grad():
            total = len(testingX0)
            totalCorrect = 0
            x0 = testingX0.to(device)
            y = testingY.to(device)
            
            outputs = model.forward(nodes, x0)
            for i in range(len(outputs)):
                if(torch.argmax(outputs[i]) == y[i]):
                    totalCorrect = totalCorrect + 1
            returnJson['testing_percent'] = round(totalCorrect/total * 100,2)
        
        returnJson['model_id'] = modelId

        return returnJson











        
# print(neuralNet.run(108842952, [[1,1],[0,0],[1,0], [0,1]]))

# inputs = [[[1],[1]],[[1],[0]],[[0],[1]],[[0],[0]],[[1],[1]],[[1],[0]],[[0],[1]],[[0],[0]],[[1],[1]],[[1],[0]],[[0],[1]],[[0],[0]],[[1],[1]],[[1],[0]],[[0],[1]],[[0],[0]],[[1],[1]],[[1],[0]],[[0],[1]],[[0],[0]],[[1],[1]],[[1],[0]],[[0],[1]],[[0],[0]]]
# outputs = [2, 1, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0,2, 1, 1, 0, 2, 1, 1, 0, 2, 1, 1, 0]

# print(neuralNet.improve(108842952,  4, 10, .01, inputs, outputs))


# print(neuralNet.create([6], 4, 10000, .01, inputs, outputs))



    





