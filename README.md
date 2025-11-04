# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: INDHUMATHI L
RegisterNumber:  212224220037
*/
```
```
import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```
## Output:

<img width="1110" height="351" alt="image" src="https://github.com/user-attachments/assets/a138fc27-3189-4658-ad22-0fd36c1ae34d" />

<img width="1258" height="638" alt="image" src="https://github.com/user-attachments/assets/f69017c6-dd83-4685-a823-309c144ef0bd" />

<img width="1596" height="410" alt="image" src="https://github.com/user-attachments/assets/2f729e7d-ee88-4d11-b7a1-54082c2ba648" />

<img width="1588" height="612" alt="image" src="https://github.com/user-attachments/assets/8d767014-9b4e-4516-bb97-9362d0c82730" />

<img width="1148" height="272" alt="image" src="https://github.com/user-attachments/assets/2908f208-8ebf-4303-8451-aef8faecca1d" />


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
