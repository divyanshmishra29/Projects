import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report


def fill_age(l1):
    m_age = l1[0]
    c_pas = l1[1]

    if pd.isnull(m_age):
        if c_pas == 1:
            return 48
        elif c_pas == 2:
            return 32
        else:
            return 25
    else:
        return m_age
    
   

train = pd.read_csv('titanic_train.csv')
print(train.head())

#data analysis

print(train.info())

sns.countplot(x = 'Survived',hue = 'Pclass',data = train)        #for survivals
plt.show()

sns.countplot(x = 'Survived',hue = 'Sex',data = train)        #for survivals are male or female
plt.show()

sns.countplot(x = 'Survived',hue = 'Pclass',data = train)        #for survivals are which class
plt.show()

train['Age'].hist()        #for age of the passenger
plt.show()

train['Fare'].hist()        #for fare of the passenger
plt.show()

#Cleaning the data
sns.boxplot(x = 'Pclass',y = 'Age',data = train)
plt.show()

train['Age'] = train[['Age','Pclass']].apply(fill_age,axis=1)
train.drop('Cabin',axis=1,inplace=True)
train.dropna(inplace = True)


#making option category in string so making dummies of string into no and concat

male =  pd.get_dummies(train['Sex'],drop_first=True)

train = pd.concat([train,male],axis = 1)
print(train.head())

train.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
print(train.head())

#algo

x = train.drop('Survived',axis=1)
y = train['Survived']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.33)

lgr = LogisticRegression()
lgr.fit(x_train,y_train)
prediction = lgr.predict(x_test)
accuracy = lgr.score(x_test,y_test)

print(accuracy)

conf_mat = confusion_matrix(y_test, prediction)
print(conf_mat)

class_rep = classification_report(y_test, prediction)
print(class_rep)


