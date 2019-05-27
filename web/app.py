from flask import Flask, render_template, json, request
from neuralNetwork.neuralNetwork import Regression, NeuralNetwork

app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/run', methods=["POST"])
def run():
    file = request.files['sourceFile']
    learning_rate = request.form['learningRate']
    max_inter = request.form['maxInter']
    algorithm = request.form['algorithm']
    percent = request.form['percent']

    #neural = NeuralNetwork(csvFile = "train.csv", test_size=percent, solver = algorithm, iterations=max_inter, lr = learning_rate)
    #regression = Regression(csvFile = file, test_size=percent)
    
    #To get predictions for train dataset from neural network:
    #neural.getPredictionTrain()
    #To get predictions from test dataset for neural network:
    #neural.getPredictionTest()
    #To get predictions from train dataset for regression:
    #regression.getPredictionTrain()
    #To get predictions from test dataset for regression:
    #regression.getPredictionTest()
    #To get prediction for given dataset (DATA in xls format)
    #neural.predictate(DATA)

    #neuralTrainY=neural.getPredictionTrain()
    #regressionTrainY=regression.getPredictionTrain()

    #TODO: add validation(optional), call machine learning module, return path to the picture (and data?)

    response = app.response_class(
        response=json.dumps("data"),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == '__main__':
   app.run(debug = True)
