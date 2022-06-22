"""
Created on Sun May 29 14:07:03 2022
@author: SUPRIYA
"""
#import pandas 
import pandas as pd 
df=pd.read_csv("C://Users/NAVEEN REDDY/Downloads/forestfires (1).csv")
df.head()
df.shape
df.isna().sum()
#due to categorical variable applying label encoding
from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df["size_category_code"] = LE.fit_transform(df["size_category"])
df[["size_category", "size_category_code"]].head(31)
pd.crosstab(df.size_category,df.size_category_code)

df.drop(["month","day","size_category"],axis=1,inplace=True)

list(df)
df.shape
x=df.iloc[:,0:28]
x.shape
y = df['size_category_code']
y.shape
# model fitting 
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
model = Sequential()
model.add(Dense(12, input_dim=28,  activation='relu'))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, validation_split=0.25, epochs=100, batch_size=15)
scores = model.evaluate(x, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#predicting y
y_pred=model.predict(x)
y_pred=y_pred>=0.5
# applying confusion matrix and accuarcy score for knowing our model predicted well or not 
import seaborn as sns
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y,y_pred)
#heat map of accuracy
sns.heatmap(cm,annot=True)
print(accuracy_score(y,y_pred))

''' inference:After predicting the model accuracy :98%,so our model predict best ,related by temperature ,high/low rain
wind blowing high/low comparing buring area is calculated '''
