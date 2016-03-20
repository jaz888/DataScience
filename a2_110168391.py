#!/usr/bin/python

import glob
import re
from happierfuntokenizing import Tokenizer
from bs4 import BeautifulSoup
from multiprocessing import Process, Queue
import os
import pickle
import csv


def run_proc(file, q):
    user_id_re = re.compile(r'(?<=/)(\d{3,8})(?=.)')
    user_id = user_id_re.findall(file)[0]
    f = open(file, 'r')
    p1 = re.compile(r'(?<=<post>)([\s\S]*?)(?=</post>)')
    posts = p1.findall(f.read())
    token_dict = {}
    for p in posts:
        tokenized = tok.tokenize(p)
        for token in tokenized:
            if token in token_dict:
                token_dict[token] += 1
            else:
                token_dict[token] = 1
    q.put(token_dict)
    print file
    #print file, len(dict[user_id])

def run_proc2(file, q):
    soup = BeautifulSoup(open(file),'xml')
    posts = soup.find_all('post')
    for post in posts:
        tokenized = tok.tokenize(post)
        for token in tokenized:
            if token in user_word_count[user_id]:
                dict[user_id][token] += 1
            else:
                dict[user_id][token] = 1
    #print file


tok = Tokenizer(preserve_case=False)
files_list = glob.glob('/Users/jieaozhu/Documents/DataScience/Differential_Topic_Analysis/samples_500/*.xml')


user_word_count = {}
posts_count = 0
words_count = 0
industry = {}
user_id_re = re.compile(r'(?<=/)(\d{3,8})(?=.)')
user_industry_re = re.compile(r'(?<=.)(\w{1,10})(?=.\w+.xml)')
post_re = re.compile(r'(?<=<post>)([\s\S]*?)(?=</post>)')
for file in files_list:
    user_id = user_id_re.findall(file)[0]
    user_industry = user_industry_re.findall(file)[0]
    if user_industry in industry:
        industry[user_industry] += 1
    else:
        industry[user_industry] = 1
    user_word_count[user_id] = {}
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
for x in industry:
    print '\t',x, industry[x]







####################################################
####################################################


# user_word_count
word_topic_prob = {}
user_word_prob = {}
user_topic_prob = {}

with open('wwbpFBtopics_condProb.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    #skip header
    next(reader)
    for row in reader:
        # term, topic_id, weight
        term = row[0]
        topic_id = int(row[1])
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

user_mention = (-1, 1.0, 1.0, 1.0)
for user in user_topic_prob:
    if 463 in user_topic_prob[user] and 963 in user_topic_prob[user] and 981 in user_topic_prob[user]:
        if user_mention[0] == -1 or user_mention[0] > user:
            user_mention = (user, user_topic_prob[user][463], user_topic_prob[user][963],user_topic_prob[user][981])
print '2. a)',user_mention[0],': 463:',user_mention[1], ', 963:', user_mention[2],', 981:',user_mention[3]
