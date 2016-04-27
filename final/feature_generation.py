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
from bs4 import BeautifulSoup
import requests
import re
import sys, traceback
import urlparse
import json
import urllib2

data = pd.read_csv('OnlineNewsPopularity_expanded.csv', sep=',',thousands=',')
# for index, row in data.iterrows():
#     print(row['url'])

youtube_regex = ( r'(https?://)?(www\.)?' '(youtube|youtu|youtube-nocookie)\.(com|be)/' '(watch\?.*?(?=v=)v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')
error_count = 0
for index, row in data.iterrows():
    if index % 150 == 0:
        print index
    if index > 20000 and row[' num_videos'] > 0 and row['youtube_views'] == 0:
        views_total = 0.0
        video_count = 0.0
        r = requests.get(row['url'])
        if r.status_code != 200:
            # error in fetching url
            print 'mashable link error'
            error_count += 1
            if error_count >= 100000:
                sys.exit(0)
        else:
            soup = BeautifulSoup(r.content,"html.parser")
            content = soup.find('section', class_='article-content')
            if content is None:
                print 'mashable link error'
                error_count += 1
                if error_count >= 100000:
                    sys.exit(0)
            iframes = soup.find_all('iframe')
            for iframe in iframes:
                src_url = iframe.get('src')
                if src_url is not None:
                    src_url = src_url.replace('embed/','watch?v=')
                    youtube_regex_match = re.match(youtube_regex, src_url)
                    if youtube_regex_match:
                        src_url_2 = src_url.rsplit('?enablejsapi', 1)[0]
                        url_data = urlparse.urlparse(src_url_2)
                        query = urlparse.parse_qs(url_data.query)
                        try:
                            video_id = query["v"][0]
                        except KeyError:
                            continue
                        info_url = 'https://www.googleapis.com/youtube/v3/videos?part=contentDetails,statistics&id='
                        info_url += video_id
                        info_url += '&key=AIzaSyAcgjLgYXlwhbkXDwDZNiGHn0baWh1JpV8'
                        j = urllib2.urlopen(info_url)
                        j = json.load(j)
                        try:
                            # print j['items'][0]['statistics']['viewCount']
                            views_total += int(j['items'][0]['statistics']['viewCount'])
                            video_count += 1.0
                        except IndexError:
                            # video is removed
                            pass
                else:
                    # not target iframe
                    pass
        data.loc[index,'youtube_views'] = views_total
        data.loc[index,'average_youtube_views'] = 0.0 if views_total == 0.0 else views_total / video_count
        data.to_csv('OnlineNewsPopularity_expanded.csv')
