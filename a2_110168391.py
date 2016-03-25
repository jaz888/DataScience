# -*- coding: utf-8 -*-
import glob
import re
from happierfuntokenizing import Tokenizer
import csv
import numpy as np
import sys
import math
import operator
import matplotlib.pyplot as plt

#
# def run_proc(file, q):
#     user_id_re = re.compile(r'(?<=/)(\d{3,8})(?=.)')
#     user_id = user_id_re.findall(file)[0]
#     f = open(file, 'r')
#     p1 = re.compile(r'(?<=<post>)([\s\S]*?)(?=</post>)')
#     posts = p1.findall(f.read())
#     token_dict = {}
#     for p in posts:
#         tokenized = tok.tokenize(p)
#         for token in tokenized:
#             if token in token_dict:
#                 token_dict[token] += 1
#             else:
#                 token_dict[token] = 1
#     q.put(token_dict)
#     # print file
#     #print file, len(dict[user_id])
#
# def run_proc2(file, q):
#     soup = BeautifulSoup(open(file),'xml')
#     posts = soup.find_all('post')
#     for post in posts:
#         tokenized = tok.tokenize(post)
#         for token in tokenized:
#             if token in user_word_count[user_id]:
#                 dict[user_id][token] += 1
#             else:
#                 dict[user_id][token] = 1
#     #print file


tok = Tokenizer(preserve_case=False)
files_list = glob.glob('/Users/jieaozhu/Documents/DataScience/Differential_Topic_Analysis/samples_500/*.xml')


user_word_count = {}
posts_count = 0
words_count = 0
industries = {}
ages = {}
genders = {}
user_industry_map = {}
sample_size = 0

user_id_re = re.compile(r'(?<=/)(\d{3,8})(?=.)')

user_age_re = re.compile(r'(?<=\.)(\d{2})(?=\.)')

user_gender_re = re.compile(r'(?<=\d\.)(\w{4,6})(?=\.\d)')

user_industry_re = re.compile(r'(?<=.)(\w{1,10})(?=.\w+.xml)')
post_re = re.compile(r'(?<=<post>)([\s\S]*?)(?=</post>)')
file_count = 1
for file in files_list:
    user_id = int(user_id_re.findall(file)[0])
    user_age = int(user_age_re.findall(file)[0])
    user_gender = 0 if user_gender_re.findall(file)[0] == 'female' else 1
    user_industry = user_industry_re.findall(file)[0]
    if user_industry in industries:
        industries[user_industry] += 1
    else:
        industries[user_industry] = 1
    user_word_count[user_id] = {}
    ages[user_id] = user_age
    genders[user_id] = user_gender
    user_industry_map[user_id] = user_industry
    sample_size += 1
    f = open(file, 'r')
    posts = post_re.findall(f.read())
    for p in posts:
        posts_count += 1
        tokenized = tok.tokenize(p)
        for token in tokenized:
            words_count += 1
            if token in user_word_count[user_id]:
                user_word_count[user_id][token] += 1
            else:
                user_word_count[user_id][token] = 1

#
# i = 0;
# q = Queue(maxsize = 0)
# for i in range(0, len(files_list), 4):
#     process1 = Process(target=run_proc, args=(files_list[i],q))
#     process1.start()
#
#     if i+1 < len(files_list):
#         process2 = Process(target=run_proc, args=(files_list[i+1],q))
#         process2.start()
#
#     if i+2 < len(files_list):
#         process3 = Process(target=run_proc, args=(files_list[i+2],q))
#         process3.start()
#
#     if i+3 < len(files_list):
#         process4 = Process(target=run_proc, args=(files_list[i+3],q))
#         process4.start()
#
#     process1.join()
#
#     if i+1 < len(files_list):
#         process2.join()
#
#     if i+2 < len(files_list):
#         process3.join()
#
#     if i+3 < len(files_list):
#         process4.join()
#
#     i += 4
#     print i


print '1. a) posts:',posts_count
print '1. b) users:',len(user_word_count)
print '1. c) words:',words_count
print '1. d):'
for x in industries:
    print '\t',x, industries[x]
print ''







####################################################
####################################################


# user_word_count
word_topic_prob = {}
user_word_prob = {}
user_topic_prob = {}
topics = {}
with open('wwbpFBtopics_condProb.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    #skip header
    next(reader)
    for row in reader:
        # term, topic_id, weight
        term = row[0]
        topic_id = int(row[1])
        if not topic_id in topics:
            topics[topic_id] = 1
        weight = float(row[2])
        if not term in word_topic_prob:
            word_topic_prob[term] = {}
        word_topic_prob[term][topic_id] = weight

for user in user_word_count:
    count_sum = 0
    user_word_prob[user] = {}
    for word in user_word_count[user]:
        count_sum += user_word_count[user][word]
    for word in user_word_count[user]:
        # skip words not in word_topic_prob
        if word in word_topic_prob:
            user_word_prob[user][word] = float(user_word_count[user][word]) / float(count_sum)

for user in user_word_prob:
    user_topic_prob[user] = {}
    for word in user_word_prob[user]:
        for topic in word_topic_prob[word]:
            if not topic in user_topic_prob[user]:
                user_topic_prob[user][topic] = 0.0
            user_topic_prob[user][topic] += word_topic_prob[word][topic]*user_word_prob[user][word]


# for user in user_word_prob:
#     print user
#     for word in user_word_prob[user]:
#         print '\t',word,user_word_prob[user][word]
#
# for user in user_topic_prob:
#     print user
#     for topic in user_topic_prob[user]:
#         print '\t',topic,user_topic_prob[user][topic]

def normal_cdf(x, mu=0,sigma=1):
    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2
def p_value(beta_hat_j, sigma_hat_j):
    if beta_hat_j > 0:
        return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
    else:
        return 2 * normal_cdf(beta_hat_j / sigma_hat_j)
user_mention_1 = (-1, 1.0, 1.0, 1.0)
for user in user_topic_prob:
    if 463 in user_topic_prob[user] and 963 in user_topic_prob[user] and 981 in user_topic_prob[user]:
        if user_mention_1[0] == -1 or user < user_mention_1[0]:
            user_mention_1 = (user, user_topic_prob[user][463], user_topic_prob[user][963],user_topic_prob[user][981])

user_mention_2 = (-1, 1.0, 1.0, 1.0)
for user in user_topic_prob:
    if 463 in user_topic_prob[user] and 963 in user_topic_prob[user] and 981 in user_topic_prob[user] and user != user_mention_1[0]:
        if user_mention_2[0] == -1 or user_mention_2[0] > user:
            user_mention_2 = (user, user_topic_prob[user][463], user_topic_prob[user][963],user_topic_prob[user][981])

user_mention_3 = (-1, 1.0, 1.0, 1.0)
for user in user_topic_prob:
    if 463 in user_topic_prob[user] and 963 in user_topic_prob[user] and 981 in user_topic_prob[user] and user != user_mention_1[0]  and user != user_mention_2[0]:
        if user_mention_3[0] == -1 or user_mention_3[0] > user:
            user_mention_3 = (user, user_topic_prob[user][463], user_topic_prob[user][963],user_topic_prob[user][981])

print '2. a)',user_mention_1[0],': 463:',user_mention_1[1], ', 963:', user_mention_1[2],', 981:',user_mention_1[3]
print '2. a)',user_mention_2[0],': 463:',user_mention_2[1], ', 963:', user_mention_2[2],', 981:',user_mention_2[3]
print '2. a)',user_mention_3[0],': 463:',user_mention_3[1], ', 963:', user_mention_3[2],', 981:',user_mention_3[3]
print ''
#########################################################
#########################################################
X = np.arange(len(user_topic_prob)*3, dtype=np.float).reshape(len(user_topic_prob), 3)
Y = np.arange(len(user_topic_prob), dtype=np.float).reshape(len(user_topic_prob), 1)
Y_row_id = 0
for user_id in user_topic_prob:
    Y[Y_row_id][0] = ages[user_id]
    Y_row_id += 1

# standardize Y
Y_std = np.std(Y, axis=0)
Y_mean = np.mean(Y,axis=0)
for row in Y:
    row[0] = (row[0] - Y_mean[0]) / Y_std[0]
# standardize Y
ageY_genderX1_topicX2_coefficients = {}
for topic_id in topics:
    X_row_id = 0
    for user_id in user_topic_prob:
        X[X_row_id][0] = 1.0
        X[X_row_id][1] = genders[user_id]
        X[X_row_id][2] = 0.0 if topic_id not in user_topic_prob[user_id] else user_topic_prob[user_id][topic_id]
        X_row_id += 1
    # standardize X
    X_std = np.std(X, axis=0)
    X_mean = np.mean(X,axis=0)
    for row in X:
        row[2] = (row[2] - X_mean[2]) / X_std[2]
    # standardize X
    res = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)),X.transpose()),Y)
    ageY_genderX1_topicX2_coefficients[topic_id] = [res[0][0], res[1][0], res[2][0]]
    #print ageY_genderX1_topicX2_coefficients[topic_id]
    # T test
    err_square_sum_Y = 0.0
    err_square_sum_X = 0.0
    for row_id in range(sample_size):
        y = Y[row_id]
        y_hat = ageY_genderX1_topicX2_coefficients[topic_id][0] + ageY_genderX1_topicX2_coefficients[topic_id][1]*X[row_id][1]+ageY_genderX1_topicX2_coefficients[topic_id][2]*X[row_id][2]
        err_square_sum_Y += math.pow(y - y_hat,2)
        err_square_sum_X += math.pow((X[row_id][2] - X_mean[2]),2)
    err_square_sum_Y = err_square_sum_Y / float(sample_size-2)
    sigma_pow2 =  err_square_sum_Y / err_square_sum_X
    #print err_square_sum_Y, err_square_sum_X
    p_val = p_value(ageY_genderX1_topicX2_coefficients[topic_id][2] ,  math.sqrt(sigma_pow2))
    ageY_genderX1_topicX2_coefficients[topic_id][0] = p_val
    if p_val > 0.025:
        ageY_genderX1_topicX2_coefficients[topic_id][2] = 0.0
    #print ageY_genderX1_topicX2_coefficients[topic_id][0],ageY_genderX1_topicX2_coefficients[topic_id][1],ageY_genderX1_topicX2_coefficients[topic_id][2], sigma
    # T test
top_beta = []
for topic_id in ageY_genderX1_topicX2_coefficients:
    if ageY_genderX1_topicX2_coefficients[topic_id][2] != 0.0:
        top_beta.append([topic_id, ageY_genderX1_topicX2_coefficients[topic_id][2],ageY_genderX1_topicX2_coefficients[topic_id][0]])
        #print topic_id, ageY_genderX1_topicX2_coefficients[topic_id][2]

top_beta = sorted(top_beta, key=lambda x: x[1], reverse=True)
for i in range(10):
    #3. a) topic_id: XXX, correlation: .XXX, p-value: .XXX, signficant after correction? (y/n)
    print '3. a) topic_id: ',top_beta[i][0],', correlation: ',top_beta[i][1],', p-value: ',top_beta[i][2],', signficant after correction: y'
print ''
for i in range(10):
    print '3. b) topic_id: ',top_beta[-1-i][0],', correlation: ',top_beta[-1-i][1],', p-value: ',top_beta[-1-i][2],', signficant after correction: y'

print ''
#########################################################
#########################################################

q4_res = {}
for industry in industries:
    if industries[industry] > 30:
        Y = np.arange(sample_size, dtype=np.float).reshape(sample_size, 1)
        Y_row_id = 0
        for user_id in user_topic_prob:
            Y[Y_row_id][0] = 1.0 if user_industry_map[user_id] == industry else 0.0
            Y_row_id += 1
        # generate matrix X
        X_row_id = 0
        X = np.arange(sample_size*4, dtype=np.float).reshape(sample_size, 4)
        for user_id in user_topic_prob:
            X[X_row_id][0] = 1.0 # beta0
            X[X_row_id][1] = 0.0 # topic
            X[X_row_id][2] = ages[user_id] # age
            X[X_row_id][3] = genders[user_id]
            X_row_id += 1

        W = np.arange(sample_size * sample_size, dtype=np.float).reshape(sample_size, sample_size)
        for i in range(sample_size):
            for j in range(sample_size):
                W[i][j] = 0.0
        Z = np.arange(sample_size, dtype=np.float).reshape(sample_size, 1)
        for topic in topics:
            if topic == 383 or topic == 1021:
                continue
            # change topic column in matrix X
            X_row_id = 0
            for user_id in user_topic_prob:
                X[X_row_id][1] = 0.0 if topic not in user_topic_prob[user_id] else user_topic_prob[user_id][topic] # topic
                X_row_id += 1

            # X standardize
            X_std = np.std(X, axis=0)
            X_mean = np.mean(X,axis=0)
            for row in X:
                row[1] = (row[1] - X_mean[1]) / X_std[1]
                row[2] = (row[2] - X_mean[2]) / X_std[2]
            # X standardize

            # calculate Pi and matrix W
            old_betas = [0.0,0.0,0.0,0.0]
            new_betas = [0.0,0.0,0.0,0.0]
            enter = False
            while (enter == False or (math.fabs(new_betas[0]-old_betas[0]) > 0.01
                or math.fabs(new_betas[1]-old_betas[1]) > 0.01
                or math.fabs(new_betas[2]-old_betas[2]) > 0.01
                or math.fabs(new_betas[3]-old_betas[3]) > 0.01)):
                enter = True
                #store old betas
                old_betas[:] = new_betas[:]
                for i in range(sample_size):
                    try:
                        e_power = math.exp(old_betas[0]+old_betas[1]*X[i][1]+old_betas[2]*X[i][2]+old_betas[3]*X[i][3])
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
            if industry not in q4_res:
                q4_res[industry] = []
            q4_res[industry].append([topic, new_betas[0][0],new_betas[1][0],new_betas[2][0],new_betas[3][0]])

    else:
        pass
for industry in q4_res:
    # 4. a) industry: XXX, topic_id: XXX, coefficient: .XXX
    q4_res[industry] = sorted(q4_res[industry], key=lambda x: x[2], reverse=True)
    for i in range(5):
        print '4. a) industry: ',industry, ', topic_id: ',q4_res[industry][i][0],', coefficient: ',q4_res[industry][i][2]
    print ''
for industry in q4_res:
    # 4. a) industry: XXX, topic_id: XXX, coefficient: .XXX
    q4_res[industry] = sorted(q4_res[industry], key=lambda x: x[2], reverse=True)
    for i in range(5):
        print '4. b) industry: ',industry, ', topic_id: ',q4_res[industry][-1-i][0],', coefficient: ',q4_res[industry][-1-i][2]
    print ''


#########################################################
#########################################################

fig = plt.figure()
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')



freqs_dict = {}

with open('2000topics.top20freqs.keys.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        # term, topic_id, weight
        topic_id = int(row[0])
        freqs_dict[topic_id] = []

        if(len(row) >= 2):
            freqs_dict[topic_id].append(row[1])
        if(len(row) >= 4):
            freqs_dict[topic_id].append(',')
            freqs_dict[topic_id].append(row[3])
        if(len(row) >= 6):
            freqs_dict[topic_id].append('\n')
            freqs_dict[topic_id].append(row[5])
        if(len(row) >= 8):
            freqs_dict[topic_id].append(',')
            freqs_dict[topic_id].append(row[7])
        freqs_dict[topic_id] = ''.join(freqs_dict[topic_id])


subplots_num = len(q4_res)
subplot_id = 1

for industry in q4_res:
    # correlation : q4_res[industry][i][2]

    x_pos_axis = 0.0
    y_pos_axis = 0.0
    x_neg_axis = 99.0
    y_neg_axis = 99.0

    ax = fig.add_subplot(subplots_num, 1, subplot_id)
    ax.set_title(industry)
    # ax.set_xlabel('xlabel')
    ax.set_ylabel('mean age')

    for i in range(5):
        x = q4_res[industry][i][2]
        topic = q4_res[industry][i][0]
        prob_ages = []
        age_sum = 0.0
        for user in user_topic_prob:
            if topic in user_topic_prob[user]:
                prob_ages.append([user_topic_prob[user][topic], ages[user]])
        prob_ages = sorted(prob_ages, key=lambda x: x[0])
        total_count = len(prob_ages)
        total_count = total_count / 4
        for j in range(total_count):
            age_sum += prob_ages[j][1]
        y = float(age_sum) / float(total_count)
        if x > x_pos_axis:
            x_pos_axis = x
        if y > y_pos_axis:
            y_pos_axis = y
        if x < x_neg_axis:
            x_neg_axis = x
        if y < y_neg_axis:
            y_neg_axis = y
        ax.text(x, y, freqs_dict[topic])

    for i in range(5):
        x = q4_res[industry][-1-i][2]
        topic = q4_res[industry][-1-i][0]
        prob_ages = []
        age_sum = 0.0
        for user in user_topic_prob:
            if topic in user_topic_prob[user]:
                prob_ages.append([user_topic_prob[user][topic], ages[user]])
        prob_ages = sorted(prob_ages, key=lambda x: x[0])
        total_count = len(prob_ages)
        total_count = total_count / 4
        for j in range(total_count):
            age_sum += prob_ages[j][1]
        y = float(age_sum) / float(total_count)
        if x > x_pos_axis:
            x_pos_axis = x
        if y > y_pos_axis:
            y_pos_axis = y
        if x < x_neg_axis:
            x_neg_axis = x
        if y < y_neg_axis:
            y_neg_axis = y
        ax.text(x, y, freqs_dict[topic],color='blue')

    subplot_id += 1
    ax.axis([x_neg_axis, x_pos_axis+1, y_neg_axis, y_pos_axis+1])

plt.show()
