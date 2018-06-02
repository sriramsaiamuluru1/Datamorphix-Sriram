import json
import pandas as pd
import numpy as np 
# from os.path import join, dirname
from watson_developer_cloud import NaturalLanguageClassifierV1

natural_language_classifier = NaturalLanguageClassifierV1(
    username='98ae2dfe-d133-4478-950c-5e39269f4d1f',
    password='ZXLPYD4CkSiy')
classes=[]
classifiers = natural_language_classifier.list()
print(json.dumps(classifiers, indent=2))
"""
# create a classifier
with open('YelpRestRev.csv', 'r') as training_data:
     print(json.dumps(natural_language_classifier.create(
		training_data=training_data, name='ReviewYelp1'), indent=2))
"""
#df.TEXT.str.decode("ISO-8859-1")
status = natural_language_classifier.status('359f41x201-nlc-49879')
print(json.dumps(status, indent=2))

df = pd.read_csv('Revtest.csv')
if status['status'] == 'Available':
	for i in range(0,len(df.TEXT),1):
		line = df.TEXT[i]
		classes.append(natural_language_classifier.classify('359f41x201-nlc-49879',line.decode("ISO-8859-1")))
with open('yelp_{}_{}.json'.format('Resto1','HOU'),'w') as f:
    json.dump(classes, f, indent=5)
"""
data = json.loads(make)
df3 = pd.read_json("yelp_Resto1_HOU.json")
df2 = pd.read_csv('Revtest.csv',encoding="ISO-8859-1")
df2['Class'] = df3['top_class']
df2.to_csv('Revtest.csv',index=False,encoding="ISO-8859-1")
"""