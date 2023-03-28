import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

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


# spliting the data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(parisdata.drop('price', axis=1),
                                                    parisdata['price'],
                                                    test_size=0.2,
                                                    random_state=42)

# create the model and fit the data
gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05,
                                 max_depth=3, random_state=42)
gbr.fit(X_train, Y_train)

# predict on test set and calculate performance metrics
YPred = gbr.predict(X_test)
mse = mean_squared_error(Y_test, YPred)
r2 = r2_score(Y_test, YPred)

print('MSE:', mse)
print('R2:', r2)
# Just to test our accuracy again
# Choose some sample data
sample_data = X_test[:10] # take the first 10 rows of X_test
actual_prices = Y_test[:10] # take the actual prices for the first 10 rows

# Use the trained model to predict the house prices
predicted_prices = gbr.predict(sample_data)

# Calculate the accuracy
accuracy = 100 * (1 - abs(actual_prices - predicted_prices).mean() / actual_prices.mean())

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy))
