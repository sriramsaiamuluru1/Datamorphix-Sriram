# use natural language toolkit
import csv
import pandas as pd
import nltk
import sys
import numpy as np
reload(sys)
sys.setdefaultencoding("utf-8")
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
# word stemmer
stemmer = LancasterStemmer()
file1reader = open("Rest_Rev.txt" ,"r")
training_data = []
for line in file1reader:
    row = line.decode("utf8", "ignore").split('|')
    training_data.append({"class" : row[2], "sentence" : row[1]})
corpus_words = {}
class_words = {}
classes = list(set([a['class'] for a in training_data]))
for c in classes:
    class_words[c] = []
for data in training_data:
    for word in nltk.word_tokenize(data['sentence']):
        stop_words = set(stopwords.words('english'))
        if word not in stop_words and word not in ["'s",".","!","?"]:
                # stem and lowercase each word
            stemmed_word = stemmer.stem(word.lower())
            # have we not seen this word already?
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1

            class_words[data['class']].extend([stemmed_word])
print ("Corpus words and counts: %s \n" % corpus_words)
print ("Class words: %s" % class_words)
sentence = "Mediocre. The first time we ate here, in January 2012, we were very pleased; everything was very tasty especially the salsa and chips. My only complaint would've been that my frijoles and rice were only lukewarm. We ate there again two weeks later and were very disappointed. Not only was the food again lukewarm but it was tasteless. Even the salsa and chips, which we mentioned to the waitress and she said yes, but they will be better tomorrow.??? No doubt, we will not be eating there again."

def calculate_class_score(sentence, class_name, show_detail=True):
    score = 0
    for word in nltk.word_tokenize(sentence):
        if stemmer.stem(word.lower()) in class_words[class_name]:
            score+=1
            if show_detail:
                print ("   match: %s" % stemmer.stem(word.lower() ))
    return score

for c in class_words.keys():
    print ("Class: %s  Score: %s \n" % (c, calculate_class_score(sentence, c)))

def calculate_class_score_commonality(sentence, class_name, show_detail=True):
    score = 0
    for word in nltk.word_tokenize(sentence):
        if stemmer.stem(word.lower()) in class_words[class_name]:
            score += (1/corpus_words[stemmer.stem(word.lower())])

            if show_detail:
                print ("   match: %s (%s)" % (stemmer.stem(word.lower()), 1 / corpus_words[stemmer.stem(word.lower())]))
    return score

for c in class_words.keys():
    print ("Class: %s  Score: %s \n" % (c, calculate_class_score_commonality(sentence, c)))

def classify(sentence):
    high_class = None
    high_score = 0
    # loop through our classes
    for c in class_words.keys():
        # calculate score of sentence for each class
        score = calculate_class_score_commonality(sentence, c, show_detail=False)
        # keep track of highest score
        if score > high_score:
            high_class = c
            high_score = score

    return high_class, high_score



test_data_df = pd.read_csv('Review_15062017_test.csv',encoding="ISO-8859-1")
appendedClass = []
for row in test_data_df.review_text:
    classif = classify(row)
    appendedClass.append(classif[0])
se = pd.Series(appendedClass)
test_data_df['Class'] = se.values
test_data_df.to_csv('Review_15062017_test.csv',index=False,encoding="utf-8")


