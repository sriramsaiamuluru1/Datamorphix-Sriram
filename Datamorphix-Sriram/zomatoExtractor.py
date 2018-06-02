#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 08:58:03 2017

@author: sameerkumar
"""

from pyzomato import Pyzomato
import os


fileExists = os.path.isfile('zomatoData.txt')
headerList = ['restaurant_id','rating','review_id','review_text','rating_color','review_time_friendly','rating_text','time_stamp','likes','comment_count','user_name','user_zomatohandle','user_foodie_level','user_level_num','foodie_color','profile_url','profile_image']
restaurantIdList = []
p = Pyzomato("a6bbe2d54ec7dd7741ef5991f9a9cafa")
results = p.search(lat="33.222063",lon="-96.972784")
restaurants = results['restaurants']
for restaurant in restaurants:
    restaurantId = 0
    restaurantId = restaurant['restaurant']['R']['res_id']
    restaurantIdList.append(restaurantId)

for restaurantId in restaurantIdList:
    reviews = p.getRestaurantReviews(restaurantId)
    userReviews = reviews['user_reviews']
    for review in userReviews:
        reviewContentList = []
        userReview = review['review']
        rating = userReview['rating']
        reviewId = userReview['id']
        reviewText = userReview['review_text'].encode('utf-8').strip()
        ratingColor = userReview['rating_color'].encode("ascii","ignore")
        reviewTimeFriendly = userReview['review_time_friendly'].encode("ascii","ignore")
        ratingText = userReview['rating_text'].encode("ascii","ignore")
        timestamp = userReview['timestamp']
        likes = userReview['likes']
        commentCount = userReview['comments_count']
        userName = userReview['user']['name'].encode("ascii","ignore")
        try:
            userZomatoHandle = userReview['user']['zomato_handle'].encode("ascii","ignore")
        except:
           userZomatoHandle = 'Not Available'
        userFoodieLevel = userReview['user']['foodie_level']
        userLevelNum = userReview['user']['foodie_level_num']
        foodieColor = userReview['user']['foodie_color'].encode("ascii","ignore")
        profileUrl = userReview['user']['profile_url']
        profileImage = userReview['user']['profile_image']
        reviewText = reviewText.replace("\r"," ")
        reviewText = reviewText.replace("\n"," ")
        reviewContentList.append(str(restaurantId))
        reviewContentList.append(str(rating))
        reviewContentList.append(str(reviewId))
        reviewContentList.append(str(reviewText))
        reviewContentList.append(str(ratingColor))
        reviewContentList.append(str(reviewTimeFriendly))
        reviewContentList.append(str(ratingText))
        reviewContentList.append(str(timestamp))
        reviewContentList.append(str(likes))
        reviewContentList.append(str(commentCount))
        reviewContentList.append(str(userName))
        reviewContentList.append(str(userZomatoHandle))
        reviewContentList.append(str(userFoodieLevel))
        reviewContentList.append(str(userLevelNum))
        reviewContentList.append(str(foodieColor))
        reviewContentList.append(str(profileUrl))
        reviewContentList.append(str(profileImage))
        if(str(fileExists) == 'False'):
            with open('zomatoData.txt','w') as dataFile:
                print 'Adding review to file'
                dataFile.write("|".join(headerList))
                dataFile.write("\n")
                dataFile.write("|".join(reviewContentList))
                dataFile.write("\n")
                fileExists = os.path.isfile('zomatoData.txt')
                
        else:
            with open('zomatoData.txt','a') as dataFile:
                print 'Appending review to file'
                dataFile.write("|".join(reviewContentList))
                dataFile.write("\n")
        