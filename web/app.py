import logging
from flask import Flask, render_template, json, request
from neuralNetwork.neuralNetwork import Regression, NeuralNetwork
 
def convert_many_single_elemen_arrays_to_one_array(array_of_arrays):
    output = []
    for i in array_of_arrays.tolist():
        output.append(i[0])
    return output

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

    #TODO: change train.csv for stream from request 
    neural = NeuralNetwork(csvFile = "train.csv", test_size=percent, solver = algorithm, iterations=max_inter, lr = learning_rate)
    regression = Regression(csvFile = "train.csv", test_size=percent)
   
    neuralTrainY=neural.getPredictionTrain()
    regressionTrainY=regression.getPredictionTrain()

    neuralTestY=neural.getPredictionTest()
    regressionTestY=regression.getPredictionTest()
 
    #calculation all vectors for plotting 

    #third plot - porownanie wyjsciowych Y dla danych testowych (wektory po kolei: y oryginalne, y neural, y regrsja)
    vectorWithOriginalYTEST = neural.y_test
    vectorWithPredictedNetworkYTEST = neuralTestY
    vectorWithPredictedRegressionYTEST = regressionTestY
    #fourth plot - porownanie bledu neural i regresji wzgledem oryginalnych danych (dla danych testowych)
    vectorWithErrorForNetworkTEST = neural.y_test-neuralTestY
    vectorWithErrorForRegressionTEST = regression.y_test-regressionTestY
    
    vectorWithOriginalYTRAIN = convert_many_single_elemen_arrays_to_one_array(vectorWithOriginalYTRAIN)
    vectorWithPredictedNetworkYTRAIN = convert_many_single_elemen_arrays_to_one_array(vectorWithPredictedNetworkYTRAIN)
    vectorWithPredictedRegressionYTRAIN = convert_many_single_elemen_arrays_to_one_array(vectorWithPredictedRegressionYTRAIN)
    vectorWithErrorForNetworkTRAIN = convert_many_single_elemen_arrays_to_one_array(vectorWithErrorForNetworkTRAIN)
    vectorWithErrorForRegressionTRAIN = convert_many_single_elemen_arrays_to_one_array(vectorWithErrorForRegressionTRAIN)
    vectorWithOriginalYTEST = convert_many_single_elemen_arrays_to_one_array(vectorWithOriginalYTEST)
    vectorWithPredictedNetworkYTEST = convert_many_single_elemen_arrays_to_one_array(vectorWithPredictedNetworkYTEST)
    vectorWithPredictedRegressionYTEST = convert_many_single_elemen_arrays_to_one_array(vectorWithPredictedRegressionYTEST)
    vectorWithErrorForNetworkTEST = convert_many_single_elemen_arrays_to_one_array(vectorWithErrorForNetworkTEST)
    vectorWithErrorForRegressionTEST = convert_many_single_elemen_arrays_to_one_array(vectorWithErrorForRegressionTEST)


    dic = {}
    dic['vectorWithOriginalYTRAIN'] = vectorWithOriginalYTRAIN
    dic['vectorWithPredictedNetworkYTRAIN'] = vectorWithPredictedNetworkYTRAIN
    dic['vectorWithPredictedRegressionYTRAIN'] = vectorWithPredictedRegressionYTRAIN
    dic['vectorWithErrorForNetworkTRAIN'] = vectorWithErrorForNetworkTRAIN
    dic['vectorWithErrorForRegressionTRAIN'] = vectorWithErrorForRegressionTRAIN
    dic['vectorWithOriginalYTEST'] = vectorWithOriginalYTEST
    dic['vectorWithPredictedNetworkYTEST'] = vectorWithPredictedNetworkYTEST
    dic['vectorWithPredictedRegressionYTEST'] = vectorWithPredictedRegressionYTEST
    dic['vectorWithErrorForNetworkTEST'] = vectorWithErrorForNetworkTEST
    dic['vectorWithErrorForRegressionTEST'] = vectorWithErrorForRegressionTEST

    response = app.response_class(
        response=json.dumps(dic),
        status=200,
        mimetype='application/json'
    )
    return response
 
if __name__ == '__main__':
   app.run(debug = True)
