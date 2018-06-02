
import sys,logging
import simplejson
import pandas as pd
import numpy as np 
# from os.path import join, dirname
from watson_developer_cloud import NaturalLanguageClassifierV1


logging.info('entered into nlc function')
#Intializing IBM Natural Language Classifier
natural_language_classifier = NaturalLanguageClassifierV1(
    username='76f01b31-8768-4816-a7e9-2c3626c237ab',
    password='voRT5NVjHAZI')
classes=[]
#Printing out existing classifiers
classifiers = natural_language_classifier.list()
print(simplejson.dumps(classifiers, indent=2))
#Print Status
status = natural_language_classifier.status('359f41x201-nlc-65743')
print(simplejson.dumps(status, indent=2))
#Reading Input Data
df = pd.read_csv('/home/bluedata/decisionengine/reviews_2017-08-08 12-55-05.txt',sep='\t')
df2=df.copy(deep=True)
#if status of the classifier is available and a review exists then extract the review and classify it and dump it into a json
if (status['status'] == 'Available' and len(df2.review_text) > 0):
    for i in range(0,len(df2.review_text),1):
	    line = df2.review_text[i]
	    classes.append(natural_language_classifier.classify('359f41x201-nlc-65743',line.decode("ISO-8859-1")))
    with open('/home/bluedata/decisionengine/yelp_{}_{}.json'.format('Resto4','HOU'),'w') as f:
	    simplejson.dump(classes, f, indent=5)
else :
    print("NO DATA AVAILABLE")