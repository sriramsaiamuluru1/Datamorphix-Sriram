import pandas as pd

df3 = pd.read_json("yelp_Resto1_HOU.json")
df2 = pd.read_csv('Revtest.csv',encoding="ISO-8859-1")
df2['Class'] = df3['top_class']
df2.to_csv('Revtest.csv',index=False,encoding="utf-8")