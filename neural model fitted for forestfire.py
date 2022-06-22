# -*- coding: utf-8 -*-
"""
Created on Sun May 29 13:03:55 2022

@author: Sai pranay
"""
#--------------------------importing the dataset----------------------------

import pandas as pd
frf = pd.read_csv("E:\\DATA_SCIENCE_ASS\\NEURAL NETWORKS\\forestfires.csv")
print(frf)
list(frf)
frf.shape
frf.info()
frf.describe()
frf.isnull().sum()

frf["size_category"].value_counts()

#-------------------- droping-----------------
frf.drop(["month","day"],axis=1,inplace = True)

frf.shape

#plot
import seaborn as sns
import matplotlib.pyplot as plt


ax = sns.boxplot(frf['area'])

#--------------------checking correlation--------------
frf.corr()


rel = frf[frf.columns[0:11]].corr()
rel

#plot

plt.figure(figsize=(10,10))
sns.heatmap(rel,annot=True)

#------------------- spiltting----------------------

x = frf.iloc[:,:28]
x

mapping = {'small': 1, 'large': 2}
frf = frf.replace(mapping)
frf

y = frf["size_category"]
y

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# --------------------- modelfitting--------------------
model = Sequential()
model.add(Dense(12, input_dim=28,  activation='relu')) #input layer
model.add(Dense(1, activation='sigmoid')) #output layer



#---------------------------- Compile model----------------------
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# ---------------------------Fit the model-----------------------------
history = model.fit(x, y, validation_split=0.25, epochs=250, batch_size=15)

# ----------------------evaluate the model--------------------------
scores = model.evaluate(x, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# summarize history for accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
