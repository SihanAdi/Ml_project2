import time
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from Linear_Regression import visibility
from SVM.svmlr import svmLr
from file import readFile
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, recall_score, precision_score


# Data processing
data = readFile.read_file("../lib/data.csv")
data.drop(["id"], axis=1, inplace=True)

print(data.describe())

# View Correlation Coefficients

# x = data.drop("diagnosis", axis=1)
# corr = x.corr()
# plt.figure(figsize=(30, 10))
# sns.heatmap(corr, annot=True)
# plt.show()
# x = data[['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
#        'concavity_mean', 'symmetry_mean', 'fractal_dimension_mean',
#        'radius_se', 'texture_se', 'smoothness_se', 'compactness_se',
#        'concavity_se', 'concave points_se', 'symmetry_se',
#        'fractal_dimension_se', 'smoothness_worst', 'compactness_worst',
#        'concavity_worst', 'symmetry_worst', 'fractal_dimension_worst']]
# x = np.asarray(x)

data.insert(loc=len(data.columns), column='intercept', value=1)
x = np.asarray(data.drop("diagnosis", axis=1))

diagnosis = data["diagnosis"].values
y = []
# m:-1, B:1
for i in diagnosis:
    if i == "M":
        y.append(-1)
    else:
        y.append(1)
y = np.asarray(y)

# visualization
# visibility.scatter_plot_SVM(data, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

min_max_scaler = preprocessing.MinMaxScaler()
# Normalize the features of the training and test data separately
x_train = min_max_scaler.fit_transform(x_train)
x_test = min_max_scaler.fit_transform(x_test)

# start time
start = time.time()
lr = svmLr(0.00001, 10000, 5000)
lr.fit(x_train, y_train)
y_pre = lr.predict(x_test)
end = time.time() - start

# Use sklearn.svm
# lr = svm.SVC()
# lr.fit(x_train,y_train)
# y_pre = lr.predict(x_test)

cm = metrics.confusion_matrix(y_test, y_pre)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])
cm_display.plot()
plt.show()
print("Accuracy : ", accuracy_score(y_test, y_pre))
print("the actual“wall-clock” time: ", end)



