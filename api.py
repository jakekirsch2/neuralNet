import flask
import json
from flask import request
from flask_cors import CORS
import neuralNet


app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

@app.route('/fully-connected/create', methods=['POST'])
def create():
    content = json.loads(request.data)
    layout = content['layout']
    return neuralNet.neuralNet.create(layout)

@app.route('/fully-connected/run', methods=['POST'])
def run():
    content = json.loads(request.data)
    modelId = content['model_id']
    x = content['inputs']

    return neuralNet.neuralNet.run(modelId, x)

@app.route('/fully-connected/train', methods=['POST'])
def trainModel():
    content = json.loads(request.data)
    modelId = content['model_id']
    batch_size = content['batch_size']
    epochs = content['epochs']
    learning_rate = content['learning_rate']
    inputs = content['inputs']
    outputs = content['outputs']

    return neuralNet.neuralNet.trainModel(modelId, batch_size, epochs, learning_rate, inputs, outputs)

@app.route('/fully-connected/test', methods=['POST'])
def test():
    content = json.loads(request.data)
    modelId = content['model_id']
    inputs = content['inputs']
    outputs = content['outputs']

    return neuralNet.neuralNet.test(modelId, inputs, outputs)
   

app.run()

