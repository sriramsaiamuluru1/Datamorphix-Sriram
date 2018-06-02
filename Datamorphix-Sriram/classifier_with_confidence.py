import logging,sys,pandas as pd
import pandas as pd
#Reading json into a dataframe
df3 = pd.read_json("/home/bluedata/decisionengine/yelp_Resto4_HOU.json")
#Creating a copy of the dataframe
df4=df3.copy(deep='True')
#Reading the input file
df5 = pd.read_csv('/home/bluedata/decisionengine/reviews_2017-08-08 12-55-05.txt',sep='\t',encoding="ISO-8859-1")
#Creating a copy of the dataframe
df6=df5.copy(deep='True')
#Extract Class name and confidence value and apppend it to the input dataframe
if (len(df6.review_text) > 0):
    class_name_list = []
    confidence_list= []
    for rows in df4.iterrows():
	   class_name_list.append(rows[1]['classes'][0]['class_name'])
	   confidence_list.append(rows[1]['classes'][0]['confidence'])
    df6['class_name'] = class_name_list
    df6["confidence"] = confidence_list
    df6.to_csv('/home/bluedata/decisionengine/cc1.txt',sep='|',index=False,encoding="ISO-8859-1")	
else:
    print(" No Classification Available")