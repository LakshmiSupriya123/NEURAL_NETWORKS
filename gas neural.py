"""
Created on Mon May 30 15:15:23 2022
@author: Supriya
"""
#import pandas
import pandas as pd 
df=pd.read_csv("C://Users/NAVEEN REDDY/Downloads/gas_turbines.csv")
df.head()
df.shape
df.isna().sum()
list(df)
df.drop([ 'AFDP', 'GTEP','TAT','CO', 'NOX','TIT','CDP'],axis=1,inplace=True)
df.corr()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(13,13))
sns.heatmap(df,annot=True)
x=df.iloc[:,0:3]
y=df["TEY"]
#model fitting
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Dense(5, input_dim=3,  activation='relu'))
model.add(Dense(1, activation='relu')) 
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
history =model.fit(x, y, validation_split=0.25, epochs=50, batch_size=10)
scores = model.evaluate(x, y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]))
history.history.keys()
# summarize history for mse
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.title('model msle')
plt.ylabel('msle')
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
