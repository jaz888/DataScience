import glob
import re
from happierfuntokenizing import Tokenizer
import csv
import numpy as np
import sys
import math
import operator

sample_size = 6

Y = np.arange(sample_size, dtype=np.float).reshape(sample_size, 1)
Y[0][0] = 0.0
Y[1][0] = 0.0
Y[2][0] = 1.0
Y[3][0] = 0.0
Y[4][0] = 0.0
Y[5][0] = 1.0

X = np.arange(sample_size*2, dtype=np.float).reshape(sample_size, 2)
X[0][0] = 1.0
X[0][1] = -1.0

X[1][0] = 1.0
X[1][1] = 0.6

X[2][0] = 1.0
X[2][1] = 0.26

X[3][0] = 1.0
X[3][1] = 0.58

X[4][0] = 1.0
X[4][1] = -1.65

X[5][0] = 1.0
X[5][1] = 1.23

W = np.arange(sample_size * sample_size, dtype=np.float).reshape(sample_size, sample_size)
for i in range(sample_size):
    for j in range(sample_size):
        W[i][j] = 0.0


Z = np.arange(sample_size, dtype=np.float).reshape(sample_size, 1)

for i in range(1):
    # calculate Pi and matrix W
    old_betas = [0.0,0.0,0.0,0.0]
    new_betas = [0.0,0.0,0.0,0.0]
    enter = False
    times = 0
    while (times <= 3):
        times += 1
        enter = True
        #store old betas
        old_betas[:] = new_betas[:]
        for i in range(sample_size):
            try:
                e_power = math.exp(old_betas[0]+old_betas[1]*X[i][1])
            except:
                #print industry, topic, 'row',i,'math range error'
                j = 0
                user = 0
                # for user in user_topic_prob:
                #     if j == i:
                #         print user, ages[user], genders[user],user_industry_map[user]
                #         break
                #     j += 1
                e_power = 99999999999.9
            p_i = e_power / (1.0 + e_power)
            #print p_i
            try:
                logit_p_i = math.log(p_i/(1.0-p_i))
            except:
                #print industry, topic, 'row',i,'divided by zero'
                j = 0
                user = 0
                # for user in user_topic_prob:
                #     if j == i:
                #         print user, ages[user], genders[user],user_industry_map[user]
                #         break
                #     j += 1
                p_i = 0.9999999999999999
                logit_p_i = 36.7368005696771
            W[i,i] = p_i*(1.0-p_i)
            Z[i][0] = logit_p_i + ((Y[i] - p_i) / (p_i * (1.0-p_i)))
        new_betas = np.dot(np.dot(np.dot(np.linalg.inv(np.dot(np.dot(X.transpose(),W), X)),X.transpose()),W),Z)
        #print new_betas[0],new_betas[1],new_betas[2],new_betas[3]
        #print new_betas[0][0],new_betas[1][0],new_betas[2][0],new_betas[3][0]
        print W
        print Z
        print new_betas[0][0], new_betas[1][0]
