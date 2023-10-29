#IDW for multi element wing

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from matplotlib import pyplot
from smt.surrogate_models import IDW

#Read Data
df=pd.read_csv("multi-element-surrogate-data.txt",delimiter="\t")

X=df.drop(["Cl","Cd","Cl/Cd"],axis=1)
y_eff=df["Cl/Cd"]

#Normalizing Data
scaler = MinMaxScaler()
X= scaler.fit_transform(X)

#Splitting Data
X_train,X_test,y_eff_train,y_eff_test=train_test_split(X, y_eff,test_size=0.2)

#Training IDW surrogate model
sm=IDW(p=X_train.shape[0])
sm.set_training_values(X_train,np.array(y_eff_train))
sm.train()

y_eff_predicted=sm.predict_values(X_test)

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