#!/usr/bin/env python

"""
    Example call:
        ./examples.py --client_id="[CLIENT ID]" --client_secret="[CLIENT SECRET]"
"""

from yelpapi import YelpAPI
import argparse
import json
import pandas as pd
business_id=[]
review_result=[]
business_results_final=[]

argparser = argparse.ArgumentParser(description='Example Yelp queries using yelpapi. Visit https://www.yelp.com/developers/v3/manage_app to get the necessary API keys.')
argparser.add_argument('--client_id', type=str, help='Yelp Fusion API client ID')
argparser.add_argument('--client_secret', type=str, help='Yelp Fusion API client secret')
args = argparser.parse_args()

yelp_api = YelpAPI(args.client_id, args.client_secret)



response = yelp_api.search_query(term='restaurant', location='houston, tx', sort_by='rating', limit=50)
with open('yelp_{}_{}.json'.format('Rest1','HOU'),'w') as f:
           json.dump(response, f, indent=5)
print('\n-------------------------------------------------------------------------\n')
for i in range(0,len(response['businesses'])):
	business_id.append(response['businesses'][i]['id'])
for i in range(0,len(business_id)):
		business_results = yelp_api.reviews_query(id=business_id[i])
		business_results_final.append(business_results)
dump1=json.dumps(business_results_final,indent=5)
df = pd.read_json(dump1)
for item in df.values:
	for review in item[0]:
		 review_result.append(repr(review['text'].replace('\n','').encode('utf-8')))
with open("YelpReviews.txt",'w') as out_file:
	for line in review_result:
		out_file.write(line+"\n")
print ("DONE")

