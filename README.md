# linear-regression
import pandas as pd
data=pd.read_csv(r"C:\Users\HARIKA\Downloads\forestfires.csv")

print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder##for limited categories
le=LabelEncoder()
print('before',data['month'])
data['month']=le.fit_transform(data['month'])
print('after',data['month'])##month converted to numbers like 0,1,2,...
data['day']=le.fit_transform(data['day'])

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=52)

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)

from sklearn.metrics import mean_squared_error
error=mean_squared_error(ytest,ypred)
print('LR ERROR',error**0.5)
