import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation, linear_model

#Already Specified Train data
dfi = pd.read_csv('/home/bluedata/decisionengine/Encoded_Classified_1.txt', sep  = '|', encoding = 'ISO-8859-1')
#Test Data input
dfi_test = pd.read_csv('/home/bluedata/decisionengine/ec1.txt', sep  = '\t', encoding = 'ISO-8859-1')
#Extracting the headers of test data
input_list=list(dfi_test)
print input_list
corr = dfi.corr()
#sns.heatmap(corr,xticklabels=corr.columns,yticklabels=corr.columns)
#Intialixating the column values on which the regression should happen
feature_cols = ['likes','comment_count','user_level_num','Average','Avoid!','Blah!','Good Enough','Great!','Insane!','Not rated','Very Bad','Well...','Big Foodie','Connoisseur'         ,'Foodie','Super Foodie','Bad Ambience','Bad Food','Bad Service','Good Ambience','Good Food','Good Service','Not Worthy','binarized_rating_text','binarized_user_foodie_level','binarized_class_name']
#Creating a list which would contain the common columns that we have in our test data and the columns intialized above
feature_cols_1 = list(set(input_list).intersection(feature_cols))
print feature_cols_1
#Spliting the data into test and train
X_train = dfi[:-1]
print len(X_train)
X_test  = dfi_test[0:]
print len(X_test)
y_train = dfi.confidence[:-1]
print len(y_train)
y_test  = dfi_test.confidence[0:]
print len(y_test)

X = X_train[feature_cols_1]
y = y_train
Xtest = X_test[feature_cols_1]
#Creating the regression model 
regr = linear_model.Lasso(alpha=0.0000000001, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
#Running the regression model
regr.fit(X, y)
#K-fold Cross Validation performed
shuffle = cross_validation.KFold(len(X), n_folds=10, shuffle=True, random_state=0)
scores = cross_validation.cross_val_score(regr, X, y ,cv=shuffle)
print("Accuracy: %.3f%% (%.3f%%)") % (scores.mean()*100.0, scores.std()*100.0)
#printing the intercept,beta values(regression coefficients),Mean Squared Error,Predicted Values, R2 Value
print regr.intercept_
print (regr.coef_)
print mean_squared_error(regr.predict(Xtest), y_test)**0.5
print regr.predict(Xtest)
print regr.score(X,y)
#Appending the scores to the input dataframe
se = pd.Series(regr.predict(Xtest))
dfi_test['score'] = se.values
print dfi_test
dfi_test.to_csv('/home/bluedata/decisionengine/Final_Output.txt',sep='|',index=False,encoding="ISO-8859-1")	
