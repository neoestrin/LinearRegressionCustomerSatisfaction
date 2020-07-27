import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

cars = pd.read_csv('/data/training/Cars.csv') #-->read the csv file
cars1 = cars.select_dtypes(exclude=['object']) #-->exclude object
imputer = SimpleImputer(missing_values=np.nan,strategy ='median') #-->use simple imputer and use median stratergy
imputer.fit(cars1) #-->fit the cars1
imputed_data = imputer.transform(cars1) #transform the cars1
new = pd.DataFrame(imputed_data,columns = cars1.columns)
mean_enginesize = new.enginesize.mean()
mean_enginesize = round(mean_enginesize,3)
corr1 = new.enginesize.corr(new.price)
corr1 = round(corr1,3)
out1 =[]
out1.append(mean_enginesize)
out1.append(corr1)
out1 = pd.DataFrame(out1)
out1.to_csv("/code/output/output1.csv",header=False,index=None)
split_names = cars['carname'].str.split(" ").str[0]
split_names = pd.DataFrame(split_names,columns=['carname'])

split_names = split_names.replace("toyouta","toyota") #-->replacing toyouta with toyota
count = split_names[(split_names["carname"]=="toyota")].count()
count.to_csv("/code/output/output2.csv",header=False,index=None)
split_names = pd.get_dummies(split_names)
new_df = pd.concat([new,split_names],axis=1)
new_df.iloc[-2:].to_csv("/code/output/output3.csv",index=None)

x = new_df.drop("price",axis=1)
y = new_df[["price"]]
#Importing required Libraries:

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_intercept = lr.intercept_
y_intercept = np.round(y_intercept, 3)

y_pred = lr.predict(x_test)

score = r2_score(y_test, y_pred)
round_sc = round(score, 3)
print(score)

#Lasso Regression:

model_lasso = Lasso(alpha=0.01)
model_lasso.fit(x_train, y_train) 

y_pred_l = model_lasso.predict(x_test)
l_score = r2_score(y_test, y_pred_l)
l_round = round(l_score, 3)

#Ridge Regression:

rr = Ridge(alpha=0.01)
rr.fit(x_train, y_train) 

y_pred_r = rr.predict(x_test)
r_score = r2_score(y_test, y_pred_r)
r_round = round(r_score, 3)

#Output4:

l4 = [round_sc, y_intercept[0]]
o4 = pd.DataFrame(l4)
o4.to_csv("/code/output/output4.csv", index = False, header = False)

#Output5:

l5 = [r_round, l_round]
o5 = pd.DataFrame(l5)
o5.to_csv("/code/output/output5.csv", index = False, header = False)