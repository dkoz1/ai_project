import logging
from flask import Flask, render_template, json, request
from neuralNetwork.neuralNetwork import Regression, NeuralNetwork
 
app = Flask(__name__)
 
@app.route('/')
def index():
   return render_template('index.html')
 
@app.route('/run', methods=["POST"])
def run():
    file = request.files['sourceFile']
    learning_rate = float(request.form['learningRate'])
    max_inter = int(request.form['maxInter'])
    algorithm = request.form['algorithm']
    percent = float(request.form['percent'])/100
   
    #TODO: add validation(optional)

    app.logger.info(learning_rate)
    app.logger.info(max_inter)
    app.logger.info(algorithm)
    app.logger.info(percent)
    app.logger.info(file)
   
   #TODO: change train.csv for stream from request 
    neural = NeuralNetwork(csvFile = "train.csv", test_size=percent, solver = algorithm, iterations=max_inter, lr = learning_rate)
    regression = Regression(csvFile = "train.csv", test_size=percent)

 
   
    neuralTrainY=neural.getPredictionTrain()
    regressionTrainY=regression.getPredictionTrain()

    neuralTestY=neural.getPredictionTest()
    regressionTestY=regression.getPredictionTest()
 
    #TODO: add calculation all vectors for plotting 
 
    app.logger.info(neural.y_train-neuralTrainY)
    app.logger.info(regression.y_train-regressionTrainY)
    
    #first plot - porownanie wyjsciowych Y dla danych treningowych (wektory po kolei: y oryginalne, y neural, y regrsja)
    vectorWithOriginalYTRAIN = neural.y_train
    vectorWithPredictedNetworkYTRAIN = neuralTrainY
    vectorWithPredictedRegressionYTRAIN = regressionTrainY
    #second plot - porownanie bledu neural i regresji wzgledem oryginalnych danych (dla danych treningowych)
    vectorWithErrorForNetworkTRAIN = neural.y_train-neuralTrainY
    vectorWithErrorForRegressionTRAIN = regression.y_train-regressionTrainY

    #third plot - porownanie wyjsciowych Y dla danych testowych (wektory po kolei: y oryginalne, y neural, y regrsja)
    vectorWithOriginalYTEST = neural.y_test
    vectorWithPredictedNetworkYTEST = neuralTestY
    vectorWithPredictedRegressionYTEST = regressionTestY
    #fourth plot - porownanie bledu neural i regresji wzgledem oryginalnych danych (dla danych testowych)
    vectorWithErrorForNetworkTEST = neural.y_test-neuralTestY
    vectorWithErrorForRegressionTEST = regression.y_test-regressionTestY

    #TODO: return vectors in json  
    
    response = app.response_class(
        response=json.dumps("data"),
        status=200,
        mimetype='application/json'
    )
    return response
 
if __name__ == '__main__':
   app.run(debug = True)
