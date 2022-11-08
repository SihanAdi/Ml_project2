import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from file import readFile
from linearRegressionSGD import SGD
from sklearn import preprocessing
import visibility

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# read cvs file
data = readFile.read_file("/Users/adisihansun/Desktop/machine learning/project2/lib/kc_house_data.csv")

print(data.describe())

# visualization
# It can be seen that the correlation coefficient of
# bathrooms, sqft_living, grade, sqft_above, sqft_living15 is greater than 0.5
corr = data.corr()
corr = corr["price"]

# It can be seen from the bar chart that the highest correlation is sqft_living
visibility.correlation_bar_chart(corr)

for feature in corr[abs(corr) > 0.5].index.tolist():
    if feature != "price":
        visibility.scatter_plot(data, feature)

y = np.array(data["price"])
x = np.array(data["sqft_living"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

min_max_scaler = preprocessing.MinMaxScaler()
# Normalize the features and target values of the training and test data separately
x_train = min_max_scaler.fit_transform(x_train.reshape(-1, 1))
y_train = min_max_scaler.fit_transform(y_train.reshape(-1, 1))
x_test = min_max_scaler.fit_transform(x_test.reshape(-1, 1))
y_test = min_max_scaler.fit_transform(y_test.reshape(-1, 1))

# model training
start = time.time()
lr = SGD(-0.1, 10000)
lr.fit(x_train, y_train)
result = lr.predict(x_test)
end = time.time() - start
print("result", result)
print("b, w", lr.w)
print("the actual“wall-clock” time: ", end)

# Visuals for Training Dataset
plt.scatter(x_train, y_train, color='blue')
plt.plot(x_train, lr.predict(x_train), color='red')
plt.title ("Visuals for Training Dataset")
plt.xlabel("sqft_living")
plt.ylabel("Price")
plt.show()
# Visuals for Test DataSet
plt.scatter(x_test, y_test, color= 'red')
plt.plot(x_train, lr.predict(x_train), color = 'blue')
plt.title("Visuals for Test DataSet")
plt.xlabel("sqft_living")
plt.ylabel("Price")
plt.show()





