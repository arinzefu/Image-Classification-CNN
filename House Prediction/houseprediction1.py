import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV


parisdata = pd.read_csv("ParisHousing.csv")

print(parisdata.head())
print(parisdata.describe())
print(parisdata.shape)
print(parisdata.info)
#finding the missing data
parisdata.isnull().sum()
missingV = parisdata.isnull().sum().sort_values(ascending=False)
missingV = pd.DataFrame(data=parisdata.isnull().sum().sort_values(ascending=False), columns=['MissingValNum'])
print(missingV)
missingV['Percent'] = missingV.MissingValNum.apply(lambda x : '{:.2f}'.format(float(x)/parisdata.shape[0] * 100))
missingV = missingV[missingV.MissingValNum > 0]
print(missingV)

#since it is cleaned there is no missing value

sns.histplot(parisdata.price)
print(sns.histplot(parisdata.price))
plt.show()
#predicting the price so we drop the price column
parisdata['Logprice'] = np.log(parisdata.price)
parisdata.drop(["price"], axis=1, inplace=True)
print(parisdata.skew().sort_values(ascending=False))

# now set the target and predictors
Y = parisdata.Logprice  # target

#  only use numeric data type
parisdata_temp = parisdata.select_dtypes(include=["int64","float64"])
X = parisdata_temp.drop(["Logprice"],axis=1)  # predictors
# split the dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 3)
#linear regression next
LinReg = LinearRegression()
# fit optimal linear regression line on training data, this performs gradient descent under the hood
LinReg.fit(X_train, Y_train)

yr_hat = LinReg.predict(X_test)

# evaluate the algorithm with a test set
LinReg_score = LinReg.score(X_test, Y_test)  # train test
print("Accuracy: ", LinReg_score)

LinReg_cv = cross_val_score(LinReg, X, Y, cv = 5, scoring= 'r2')
print("Cross-validation results: ", LinReg_cv)
print("R2: ", LinReg_cv.mean())
