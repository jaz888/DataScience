import numpy as np
import scipy.stats as ss
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from math import log
from math import sqrt
from sklearn import preprocessing
import string
import sys
import random
from sklearn.decomposition import PCA

if len(sys.argv) > 2:
    data = pd.read_csv(sys.argv[1])
else:
    data = pd.read_csv('2015 CHR Analytic Data.csv', sep=',',thousands=',')


# question 1
valsColumn = [ currColumn for currColumn in data.columns  if "Value" in currColumn or "COUNTYCODE" in currColumn]
data = data[data['2011 population estimate Value'] >= 30000]
data = data[data.COUNTYCODE != 0]
data = data[valsColumn]
valsColumn = [v for v in valsColumn if "Value" in v]
data = data.dropna()
print "1. TOTAL NUMBER OF COUNTIES: %d" % data.shape[0]

# question 2
png_file = '2histogram.png'
print "2. log_paamv HISTOGRAM: %s" % png_file
if data['Premature age-adjusted mortality Value'].dtype != 'float' and data['Premature age-adjusted mortality Value'].dtype != 'int':
    data['Premature age-adjusted mortality Value'] = data['Premature age-adjusted mortality Value'].astype('float')
data['log_paamv'] = np.log(data['Premature age-adjusted mortality Value'])
# hist = data['log_paamv'].hist(bins=20)
# hist.get_figure().savefig(png_file)
# plt.show()




# question 3
REMOVED_COLUMNS = ['COUNTYCODE', 'log_paamv', 'Premature age-adjusted mortality Value', 'Premature death Value',  'Uninsured adults Value', 'Teen births Value', 'Food insecurity Value', 'Physical inactivity Value',  'Adult smoking Value', 'Injury deaths Value', 'Motor vehicle crash deaths Value', 'Drug poisoning deaths Value',  'Child mortality Value', 'Uninsured adults Value', 'Uninsured children Value']
Y_vector = data['log_paamv']
X_matrix = data
X_matrix = X_matrix.drop(REMOVED_COLUMNS, 1)
# partition
index = [x for x in range(X_matrix.shape[0])]
random.shuffle(index)
partition = X_matrix.shape[0] / 10
error = 0.0
coef = None
for i in range(10):
    start = i * partition
    end = i * partition + partition
    sub_X_testing = X_matrix.iloc[index[start:end]]
    sub_Y_testing = Y_vector.iloc[index[start:end]]
    sub_X_training = X_matrix.drop(X_matrix.index[range(start, end+1)])
    sub_Y_training = Y_vector.drop(Y_vector.index[range(start, end+1)])
    sub_X_training = preprocessing.scale(sub_X_training)
    sub_Y_training = preprocessing.scale(sub_Y_training)
    sub_X_testing = preprocessing.scale(sub_X_testing)
    sub_Y_testing = preprocessing.scale(sub_Y_testing)
    clf = linear_model.LinearRegression()
    clf.fit(sub_X_training, sub_Y_training)
    square = (clf.predict(sub_X_testing) - sub_Y_testing) ** 2
    square = square.sum() / len(square)
    error += square
error /= 10
print "3. Non-regularized Linear Regression MSE: %f" % error




pca = PCA(n_components=3)
pca.fit(preprocessing.scale(X_matrix))
print "4. Percentage variance explained of first three components: ", pca.explained_variance_ratio_


for c in range(1,44):
    print 'c======',c
    pca = PCA(n_components=c)
    pca.fit(preprocessing.scale(X_matrix))
    component_matrix = pd.DataFrame(pca.fit_transform(preprocessing.scale(X_matrix)))
    partition = component_matrix.shape[0] / 10
    error_pca = 0.0
    clf = linear_model.LinearRegression()
    for i in range(10):
        start = i * partition
        end = i * partition + partition
        sub_X_testing = component_matrix.iloc[index[start:end]]
        sub_Y_testing = Y_vector.iloc[index[start:end]]
        sub_X_training = component_matrix.drop(component_matrix.index[range(start, end+1)])
        sub_Y_training = Y_vector.drop(Y_vector.index[range(start, end+1)])
        sub_X_training = preprocessing.scale(sub_X_training)
        sub_Y_training = preprocessing.scale(sub_Y_training)
        sub_X_testing = preprocessing.scale(sub_X_testing)
        sub_Y_testing = preprocessing.scale(sub_Y_testing)
        clf.fit(sub_X_training, sub_Y_training)
        square = (clf.predict(sub_X_testing) - sub_Y_testing) ** 2
        square = square.sum() / len(square)
        error_pca += square
    error_pca /= 10
    print "5. a) principal components regression mse: %f" % error_pca

    error_l2 = 0.0
    clf = linear_model.Ridge (alpha = 0.001)
    for i in range(10):
        start = i * partition
        end = i * partition + partition
        sub_X_testing = component_matrix.iloc[index[start:end]]
        sub_Y_testing = Y_vector.iloc[index[start:end]]
        sub_X_training = component_matrix.drop(component_matrix.index[range(start, end+1)])
        sub_Y_training = Y_vector.drop(Y_vector.index[range(start, end+1)])
        sub_X_training = preprocessing.scale(sub_X_training)
        sub_Y_training = preprocessing.scale(sub_Y_training)
        sub_X_testing = preprocessing.scale(sub_X_testing)
        sub_Y_testing = preprocessing.scale(sub_Y_testing)
        clf.fit(sub_X_training, sub_Y_training)
        square = (clf.predict(sub_X_testing) - sub_Y_testing) ** 2
        square = square.sum() / len(square)
        error_l2 += square
    error_l2 /= 10
    print "5. b) L2 regularized  mse: %f" % error_l2

    error_l1 = 0.0
    clf = linear_model.Lasso(alpha = 0.001)
    for i in range(10):
        start = i * partition
        end = i * partition + partition
        sub_X_testing = component_matrix.iloc[index[start:end]]
        sub_Y_testing = Y_vector.iloc[index[start:end]]
        sub_X_training = component_matrix.drop(component_matrix.index[range(start, end+1)])
        sub_Y_training = Y_vector.drop(Y_vector.index[range(start, end+1)])
        sub_X_training = preprocessing.scale(sub_X_training)
        sub_Y_training = preprocessing.scale(sub_Y_training)
        sub_X_testing = preprocessing.scale(sub_X_testing)
        sub_Y_testing = preprocessing.scale(sub_Y_testing)
        clf.fit(sub_X_training, sub_Y_training)
        square = (clf.predict(sub_X_testing) - sub_Y_testing) ** 2
        square = square.sum() / len(square)
        error_l1 += square
    error_l1 /= 10
    print "5. c) L1 regularized  mse: %f" % error_l1
