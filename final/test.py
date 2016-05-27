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
import requests
import re
import sys, traceback
import urlparse
import json
import urllib2
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier





forest = RandomForestClassifier(n_estimators = 100)

data = pd.read_csv('OnlineNewsPopularity_expanded.csv', sep=',',thousands=',')
error_count = 0

REMOVED_COLUMNS = ['average_youtube_views','youtube_views',' num_imgs',' num_videos','label','log_share','url',' shares',' data_channel_is_lifestyle',' data_channel_is_entertainment',' data_channel_is_bus',' data_channel_is_socmed',' data_channel_is_tech',' data_channel_is_world',' weekday_is_monday',' weekday_is_tuesday',' weekday_is_wednesday',' weekday_is_thursday',' weekday_is_friday',' weekday_is_saturday',' weekday_is_sunday',' is_weekend']
X = data.drop(REMOVED_COLUMNS,1)
Y = data['label']

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X_normal, Y, test_size=0.4, random_state=1)

forest = forest.fit(X_train,Y_train)
output = forest.predict(X_train)
