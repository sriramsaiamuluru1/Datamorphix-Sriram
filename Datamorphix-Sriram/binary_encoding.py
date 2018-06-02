
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
logging.info('performing binary encoding')
#Reading Input File
other_CSV = pd.read_csv('/home/bluedata/decisionengine/reviews_2017-08-08 12-55-05.txt', sep  = '\t', encoding = 'ISO-8859-1')
other_CSV_0 = other_CSV.copy(deep="True")
#Try block contains the code that would perfom binarization provided that there are 1 or 3 and more categorical values
#This is a special case of label binarizer fuction pls referto my post on stackoverflow ,https://stackoverflow.com/questions/45577121/labelbinarizer-not-working-for-2-categorical-values
try:
    lb_style= LabelBinarizer()
    rating_text=lb_style.fit_transform(other_CSV["rating_text"])
    rating_text_df=pd.DataFrame(rating_text,columns=lb_style.classes_)
    other_CSV_1=other_CSV.join(rating_text_df)
    print other_CSV_1
except:
    lb_style= LabelBinarizer()
    rating_text=lb_style.fit_transform(other_CSV["rating_text"])
    rating_text_df=pd.DataFrame(rating_text,columns=['binarized_rating_text'])
    other_CSV_1=other_CSV.join(rating_text_df)
    print other_CSV_1

        
try :
        print("Entered in to try")
        lb_style = LabelBinarizer()
        user_foodie_level = lb_style.fit_transform(other_CSV["user_foodie_level"])
        user_foodie_level_df = pd.DataFrame(user_foodie_level, columns=lb_style.classes_)
        other_CSV_2 = other_CSV_1.join(user_foodie_level_df)
        print other_CSV_2
        
except :
    
        lb_style = LabelBinarizer()
        user_foodie_level = lb_style.fit_transform(other_CSV["user_foodie_level"])
        user_foodie_level_df = pd.DataFrame(user_foodie_level, columns=["binarized_user_foodie_level"])
        other_CSV_2 = other_CSV_1.join(user_foodie_level_df)
        print other_CSV_2

try:        
        lb_style = LabelBinarizer()
        class_name = lb_style.fit_transform(other_CSV["class_name"])
        class_name_df = pd.DataFrame(class_name, columns=lb_style.classes_)
        other_CSV_3 = other_CSV_2.join(class_name_df)
except:
        lb_style = LabelBinarizer()
        class_name = lb_style.fit_transform(other_CSV["class_name"])
        class_name_df = pd.DataFrame(class_name, columns=['binarized_class_name'])
        other_CSV_3 = other_CSV_2.join(class_name_df)        

other_CSV_3.to_csv("/home/bluedata/decisionengine/ec1.txt",sep = "|", index=False, encoding = 'utf-8')