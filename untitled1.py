# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:59:16 2019

@author: Divyansh Mishra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv('advertising.csv')
x=dataset.drop(['Age','Area Income','Ad Topic Line','City','Country','Timestamp','Clicked on Ad'],axis=1)
y=dataset['Clicked on Ad']
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,y_train)
prediction = log.predict(X_test)
accuracy = log.score(X_test,y_test)

print(accuracy)
