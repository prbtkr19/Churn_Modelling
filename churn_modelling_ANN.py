#part-1 -Data Preprocessing
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the datasets
dataset=pd.read_csv("F:\P16-Deep-Learning-AZ\Artificial_Neural_Networks\Churn_Modelling.csv")
#creating matrces of features
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values
#encoding categorical data
labelencoder_x_1=LabelEncoder()
x[:,1]=labelencoder_x_1.fit_transform(x[:,2])
labelencoder_x_2=LabelEncoder()
x[:,2]=labelencoder_x_2.fit_transform(x[:,2])
#creating dummy variables of country
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
#avoiding dummy variable trap
x=x[:,1:]

#splitting the dataset into training and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_text=sc.transform(x_test)
#part-2 -let's make ANN
#import keras library and packages
import keras
#import modules
from keras.models import Sequential
from keras.layers import Dense
from keras .layers import Dropout
#initializing the ANN
#defining sequemce of layers
classifier=Sequential()
#adding the input layer and first hidden layer with dropout
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu',input_dim=11))
classifier.add(Dropout(p=0.1))
#add second hidden layer
classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
classifier.add(Dropout(p=0.1))
#Adding the output layer
classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))
#compile ANN
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#fit the ANN to the training set
classifier.fit(x_train,y_train,batch_size=5,nb_epoch=100)
#part 3-making prediction and evaluate the model
#predict the test set results
y_pred=classifier.predict(x_test)
#applying thresholding
y_pred=(y_pred>0.5)
#predicting a single new observation
new_prediction=classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction=(new_prediction>0.5)
#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
#part 4- evaluating,improving and turing the ANN
#evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
     classifier=Sequential()
     classifier.add(Dense(output_dim = 6, init='uniform', activation='relu',input_dim=11))
     classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
     classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))
     classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
     return classifier
classifier=KerasClassifier(build_fn=build_classifier,batch_size=10,nb_epoch=100)
accuracies=cross_val_score(estimator=classifier, x=x_train,y = y_train,cv=10,n_jobs=-1)


 #improve the ANN
 #dropout regularization to reduce overfitting
 #turing the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
     classifier=Sequential()
     classifier.add(Dense(output_dim = 6, init='uniform', activation='relu',input_dim=11))
     classifier.add(Dense(output_dim = 6, init='uniform', activation='relu'))
     classifier.add(Dense(output_dim = 1, init='uniform', activation='sigmoid'))
     classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
     return classifier
classifier=KerasClassifier(build_fn=build_classifie)
parameters={'batch_size:[25,32],
             'nb_epoch':[100,500],
             'optimizer':['adam','rmsprop']}
grid_search=GridSearchCV(estimator=classifier,
                          param_grid=parameters,
                          scoring='accuracy',
                           cv=10)
grid_search=grid_search.fit(x_train,y_train)
best_parameters=grid_search.best_params_
best_accuracy=grid_search.best_score_



 
    