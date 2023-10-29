#ANN Surrogate model for multi elemet wing

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from matplotlib import pyplot

#Read Data
df=pd.read_csv("multi-element-surrogate-data.txt",delimiter="\t")

X=df.drop(["Cl","Cd","Cl/Cd"],axis=1)
y_eff=df["Cl/Cd"]

#Normalizing Data
scaler = MinMaxScaler()
X= scaler.fit_transform(X)


#Splitting Data
X_train,X_test,y_eff_train,y_eff_test=train_test_split(X, y_eff,test_size=0.1)

#ANN
model = keras.Sequential([
    keras.layers.Dense(160, input_shape=(4,), activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(80, activation='relu'),
    keras.layers.Dense(40, activation='relu'),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(1)
])


model.compile(optimizer='adam',
              loss='mean_absolute_error',
              metrics=['accuracy'])

model.fit(X_train, y_eff_train, epochs=2000)

model.evaluate(X_test,y_eff_test)

y_eff_predicted=model.predict(X_test)


# Calculate R-squared score for the predictions
r2 = r2_score(y_eff_test, y_eff_predicted)
print(f"R-squared score: {r2}")


#plotting r2 score
lw=2
color='black'
pyplot.figure(figsize=(11,7),dpi=600,facecolor=color)
pyplot.rcParams["font.family"] = "Times New Roman"
pyplot.rcParams["axes.linewidth"] = 2
ax=pyplot.axes()
ax.set_facecolor(color)
ax.spines['left'].set_color('white')
ax.spines['bottom'].set_color('white')
ax.tick_params(axis='both', colors='white',size=8,labelsize=17,direction='inout')

pyplot.text(10, 60, '$R^2$ score = ' + str(round(r2,3)), fontsize = 17,color='white')
pyplot.plot(y_eff_test,y_eff_test,color='yellow')
pyplot.scatter(y_eff_predicted,y_eff_test,color='red',marker='o',s=50)
pyplot.xlabel('Predicted $C_l/C_d [-]$',color='white',fontsize=20)
pyplot.ylabel('Actual $C_l/C_d [-]$',color='white',fontsize=20)

