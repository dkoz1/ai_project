import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from matplotlib.pyplot import scatter


class Regression(object):
    def __init__(self, inputName = "train.csv", target='medv', checkpoint=250):
        # inputName - where to take data
        # target - which data is our Y(Taret variable)
        # checkpoint - set the checkpoint for train-test split
        self.df = pd.read_csv(inputName)  # read data
        self.X_train = self.df.drop([target], axis=1)[0:checkpoint]  # drop the target variable
        self.X_test = self.df.drop([target], axis=1)[checkpoint:]  # drop the target variable
        self.Y_train = self.df[target][0:checkpoint]  # only the target variable as result
        self.Y_test = self.df[target][checkpoint:]  # only the target variable as result
        self.network = LinearRegression()
        self.network.fit(self.X_train, self.Y_train)  # fit the data
    def getPredictionTrain(self, printStats = True):
        pred_train=self.network.predict(self.X_train) #get predictions for the test data
        if printStats == True:
            print(r2_score(self.Y_train, pred_train))  # higher-better, values from 0 to 1, result around 0.76
            print(mean_squared_error(self.Y_train, pred_train))   # lower-better, results around 18
        return pred_train
    def getPredictionTest(self, printStats = True):
        pred_test = self.network.predict(self.X_test) #get predictions for the train data
        if printStats == True:
            print(r2_score(self.Y_test, pred_test))  # higher-better, values from 0 to 1, results close to 0
            print(mean_squared_error(self.Y_test, pred_test))  # lower-better, results around 217
        return pred_test
    def predictate(self, data):
        return self.network.predict(data)


class NeuralNetwork(object):

    def __init__(self, inputName = "train.csv", target='medv', checkpoint=250):
        # inputName - where to take data
        # target - which data is our Y(Taret variable)
        # checkpoint - set the checkpoint for train-test split
        self.df = pd.read_csv(inputName)  # read data
        self.X_train=self.df.drop([target],axis=1)[0:checkpoint] #drop the target variable
        self.X_test=self.df.drop([target],axis=1)[checkpoint:] #drop the target variable
        self.Y_train=self.df[target][0:checkpoint] #only the target variable as result
        self.Y_test=self.df[target][checkpoint:] #only the target variable as result
        self.network = MLPRegressor(hidden_layer_sizes=(100,100,100,100,100), #create the network 5 layers, 100 neurons each
                                     activation='relu', #state-of-art activation function
                                     solver='adam', #optimization algorithm
                                     max_iter=1000,
                                     learning_rate_init=0.0001)
        self.network.fit(self.X_train,self.Y_train) #fit the data

    def getPredictionTrain(self, printStats = True):
        pred_train=self.network.predict(self.X_train) #get predictions for the test data
        if printStats == True:
            print(r2_score(self.Y_train, pred_train))  # higher-better, values from 0 to 1, result around 0.76
            print(mean_squared_error(self.Y_train, pred_train))  # lower-better, results around 17
        return pred_train
    def getPredictionTest(self, printStats = True):
        pred_test = self.network.predict(self.X_test) #get predictions for the train data
        if printStats == True:
            print(r2_score(self.Y_test, pred_test))  # higher-better, values from 0 to 1, results close to 0
            print(mean_squared_error(self.Y_test, pred_test))  # lower-better, results around 43
        return pred_test
    def predictate(self, data):
        return self.network.predict(data)

neural = NeuralNetwork()
regression = Regression()

neuralTrainY=neural.getPredictionTrain()
regressionTrainY=regression.getPredictionTrain()

plt.plot(neural.Y_train-neuralTrainY,'ob')
plt.plot(regression.Y_train-regressionTrainY,'or')
plt.show()

#scatter(range(250),y=Y_train-pred_train)
#plt.plot(pred_test-,'o')
#plt.show()


