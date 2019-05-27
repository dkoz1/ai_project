import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from matplotlib.pyplot import scatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Regression(object):
    def __init__(self, inputName = "train.csv", target='medv', test_size=0.2, random_state=42):
        self.df = pd.read_csv(inputName)  # read data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df.drop(['medv'], axis=1).values, self.df['medv'].values, test_size=test_size, random_state=random_state)
        self.y_train=self.y_train.reshape(-1,1)
        self.y_test=self.y_test.reshape(-1,1)
        self.scalerX = StandardScaler().fit(self.X_train) #create a scaler which will transform the data so it has mean=0 and variation=1
        self.scalerY = StandardScaler().fit(self.y_train)
        self.X_train_scaled= self.scalerX.transform(self.X_train)
        self.X_test_scaled = self.scalerX.transform(self.X_test)
        self.y_train_scaled = self.scalerY.transform(self.y_train)
        self.y_test_scaled = self.scalerY.transform(self.y_test)
        self.regr = LinearRegression()
        self.regr.fit(self.X_train_scaled,self.y_train_scaled) #fit the data #fit the data
    def getPredictionTrain(self, printStats = True):
        pred_train_nn=self.scalerY.inverse_transform(self.regr.predict(self.X_train_scaled))
        if printStats == True:
            print("R2_train: " + str(r2_score(self.y_train, pred_train_nn))) #higher-better, values from 0 to 1, result around 0.98
            print("MSE_train: " + str(mean_squared_error(self.y_train, pred_train_nn))) #lower-better, results around 1.3
        return pred_train_nn
    def getPredictionTest(self, printStats = True):
        pred_test_nn=self.scalerY.inverse_transform(self.regr.predict(self.X_test_scaled)) #get predictions for the test dataa
        if printStats == True:
            print("R2_test: " + str(r2_score(self.y_test, pred_test_nn))) #higher-better, values from 0 to 1, result around 0.85
            print("MSE_test: " + str(mean_squared_error(self.y_test, pred_test_nn))) #lower-better, results around 13
        return pred_test_nn
    def predictate(self, data):
	    return self.scalerY.inverse_transform(self.regr.predict(self.scalerX.transform(data)))


class NeuralNetwork(object):
    def __init__(self, inputName = "train.csv", target='medv', test_size=0.2, random_state=42):
        self.df = pd.read_csv(inputName)  # read data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df.drop(['medv'],axis=1).values, self.df['medv'].values, test_size=test_size, random_state=random_state)
        self.y_train=self.y_train.reshape(-1,1)
        self.y_test=self.y_test.reshape(-1,1)
        self.scalerX = StandardScaler().fit(self.X_train) #create a scaler which will transform the data so it has mean=0 and variation=1
        self.scalerY = StandardScaler().fit(self.y_train)
        self.X_train_scaled= self.scalerX.transform(self.X_train)
        self.X_test_scaled = self.scalerX.transform(self.X_test)
        self.y_train_scaled = self.scalerY.transform(self.y_train)
        self.y_test_scaled = self.scalerY.transform(self.y_test)
        self.network = MLPRegressor(hidden_layer_sizes=(100,100,100,100,100), #create the network 5 layers, 100 neurons each
                                             activation='relu', #state-of-art activation function
                                             solver='adam', #optimization algorithm
                                             max_iter=1000,
                                             learning_rate_init=0.0001)
        self.network.fit(self.X_train_scaled,self.y_train_scaled) #fit the data #fit the data

    def getPredictionTrain(self, printStats = True):
        pred_train_nn=self.scalerY.inverse_transform(self.network.predict(self.X_train_scaled))
        if printStats == True:
            print("R2_train: " + str(r2_score(self.y_train, pred_train_nn))) #higher-better, values from 0 to 1, result around 0.98
            print("MSE_train: " + str(mean_squared_error(self.y_train, pred_train_nn))) #lower-better, results around 1.3
        return pred_train_nn
    def getPredictionTest(self, printStats = True):
        pred_test_nn=self.scalerY.inverse_transform(self.network.predict(self.X_test_scaled)) #get predictions for the test dataa
        if printStats == True:
            print("R2_test: " + str(r2_score(self.y_test, pred_test_nn))) #higher-better, values from 0 to 1, result around 0.85
            print("MSE_test: " + str(mean_squared_error(self.y_test, pred_test_nn))) #lower-better, results around 13
        return pred_test_nn
    def predictate(self, data):
	    return self.scalerY.inverse_transform(self.network.predict(self.scalerX.transform(data)))

neural = NeuralNetwork()
regression = Regression()

neuralTrainY=neural.getPredictionTrain()
regressionTrainY=regression.getPredictionTrain()

plt.plot(neural.y_train-neuralTrainY,'ob')
plt.plot(regression.y_train-regressionTrainY,'or')
plt.show()

#scatter(range(250),y=Y_train-pred_train)
#plt.plot(pred_test-,'o')
#plt.show()


