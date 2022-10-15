import os
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from scipy.sparse import hstack
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

def feature_based_combination(train, test, c, x1, y1, x2, y2 ):

  emotions = [ "Love", "Joy", "Surprise", "Anger", "Sadness", "Fear"]

  fin = []

  temp = []

  f1_dic = dict()

  scores = []
  test_targ_id = []
  test_targ_id = test["ID"].tolist()

  # temp.append( test_["ID"].tolist() )

  for emotion in emotions:
    trainData = train['Data']
    trainLabel = train[emotion]
    testData = test['Data']
    testLabel = test[emotion].tolist()


    # tfidf_vect_ngram = TfidfVectorizer(analyzer='char', ngram_range=(7,7), max_features=300000, max_df = 0.99)
    word_tfidf_vect_ngram = TfidfVectorizer(analyzer="word", ngram_range=(x1,y1), tokenizer=lambda x: x.split())

    word_fit = word_tfidf_vect_ngram.fit(trainData)
    feature_names_words = word_tfidf_vect_ngram.get_feature_names()
    xtrain_words = word_tfidf_vect_ngram.transform(trainData) 
    xtest_words = word_tfidf_vect_ngram.transform(testData)


    char_tfidf_vect_ngram = TfidfVectorizer(analyzer="char", ngram_range=(x2,y2), tokenizer=lambda x: x.split())

    char_tfidf_vect_ngram.fit(trainData)
    feature_names_char = char_tfidf_vect_ngram.get_feature_names()
    xtrain_char = char_tfidf_vect_ngram.transform(trainData) 
    xtest_char = char_tfidf_vect_ngram.transform(testData)

    tfidf_matrix_word_char_train =  hstack((xtrain_words, xtrain_char))
    tfidf_matrix_word_char_test =  hstack((xtest_words, xtest_char))
    

    linear = LinearSVC(C = c, penalty='l2', loss = 'squared_hinge') #squared_

    linear.fit(tfidf_matrix_word_char_train, trainLabel)

    pred_test = linear.predict(tfidf_matrix_word_char_test)


    oneCount = 0
    for i in range(len(pred_test)):
      if(pred_test[i] == testLabel[i] and testLabel[i] == 1):
        oneCount += 1
    
    # print("oneCount: ",oneCount)

    pred_tem_test = pred_test.tolist()
    # temp.append( pred_tem_test )
    # temp.append( testLabel )


    arr = confusion_matrix(testLabel, pred_tem_test)
    # print(arr)

    f1 = f1_score( testLabel, pred_tem_test )

    scores.append( round(f1 * 100, 2) )

    f1_dic[emotion] = round(f1 * 100, 2)
  
  print(f1_dic)
  #print(scores)
  print( "Macro Average F1-score: ", float('{0:.2f}'.format(sum(scores)/len(scores))) )


def feature_based(train, test, c, x1, y1, gram):

    emotions = [ "Love", "Joy", "Surprise", "Anger", "Sadness", "Fear"]

    f1_dic = dict()

    scores = []

    for emotion in emotions:
      trainData = train['Data']
      trainLabel = train[emotion]
      testData = test['Data']
      testLabel = test[emotion].tolist()



      # tfidf_vect_ngram = TfidfVectorizer(analyzer='char', ngram_range=(7,7), max_features=300000, max_df = 0.99)
      tfidf_vect_ngram = TfidfVectorizer(analyzer=gram, ngram_range=(x1,y1), tokenizer=lambda x: x.split())

      tfidf_vect_ngram.fit(trainData)
      feature_names = tfidf_vect_ngram.get_feature_names()
      xtrain =  tfidf_vect_ngram.transform(trainData) 
      xtest =  tfidf_vect_ngram.transform(testData)


      linear = LinearSVC(C = c, penalty='l2', loss = 'squared_hinge') #squared_

      linear.fit(xtrain, trainLabel)

      pred_test = linear.predict(xtest)
      oneCount = 0
      for i in range(len(pred_test)):
          if(pred_test[i] == testLabel[i] and testLabel[i] == 1):
            oneCount += 1

      # print("oneCount: ",oneCount)

      pred_tem_test = pred_test.tolist()

      f1 = f1_score( testLabel, pred_tem_test )

      scores.append( round(f1 * 100, 2) )

      f1_dic[emotion] = round(f1 * 100, 2)

    print(f1_dic)
    #print(scores)
    print( "Macro Average F1-score: ", float('{0:.2f}'.format(sum(scores)/len(scores))) )



if __name__ == '__main__':

  path_parent = os.path.dirname( os.getcwd() )
  os.chdir( path_parent )
  df_train = pd.read_csv( os.getcwd() + "/EmoNoBa Dataset/Final_Train.csv" )
  df_val =  pd.read_csv( os.getcwd() + "/EmoNoBa Dataset/Final_Val.csv" )
  df_test = pd.read_csv( os.getcwd() + "/EmoNoBa Dataset/Final_Test.csv" )

  model_name = input("Please Specify the Model Name: ")

  if model_name == "W1":
    feature_based(df_train, df_test, c = 10, x1 = 1, y1 = 1, gram = "word" )

  elif model_name == "W2":
    feature_based(df_train, df_test, c = 10, x1 = 2, y1 = 2, gram = "word" )

  elif model_name == "W3":
    feature_based(df_train, df_test, c = 10, x1 = 3, y1 = 3, gram = "word" )

  elif model_name == "W4":
    feature_based(df_train, df_test, c = 10, x1 = 4, y1 = 4, gram = "word" )
  
  elif model_name == "W1+W2":
    feature_based(df_train, df_test, c = 10, x1 = 1, y1 = 2, gram = "word" )

  elif model_name == "W1+W2+W3":
    feature_based(df_train, df_test, c = 10, x1 = 1, y1 = 3, gram = "word" )
  
  elif model_name ==  "W1+W2+W3+W4":
    feature_based(df_train, df_test, c = 10, x1 = 1, y1 = 4, gram = "word" )
  
  elif model_name == "C2":
    feature_based(df_train, df_test, c = 10, x1 = 2, y1 = 2, gram = "char" )

  elif model_name == "C3":
    feature_based(df_train, df_test, c = 10, x1 = 3, y1 = 3, gram = "char" )

  elif model_name == "C4":
    feature_based(df_train, df_test, c = 10, x1 = 4, y1 = 4, gram = "char" )
  
  elif model_name == "C5":
    feature_based(df_train, df_test, c = 10, x1 = 5, y1 = 5, gram = "char" )

  elif model_name == "C1+C2+C3":
    feature_based(df_train, df_test, c = 1, x1 = 1, y1 = 3, gram = "char" )
    
  elif model_name == "C1+C2+C3+C4":
    feature_based(df_train, df_test, c = 10, x1 = 1, y1 = 4, gram = "char" )
  
  elif model_name == "C1+C2+C3+C4+C5":
    feature_based(df_train, df_test, c = 10, x1 = 1, y1 = 5, gram = "char" )
  
  elif model_name == "W1+C1+C2+C3+C4+C5":
    feature_based_combination( df_train, df_test, c = 10, x1 = 1, y1 = 1, x2 = 1, y2 = 5 )

  elif model_name == "W1+W2+W3+C1+C2+C3":
    feature_based_combination( df_train, df_test, c = 10, x1 = 1, y1 = 3, x2 = 1, y2 = 3 )

  elif model_name == "W1+W2+W3+W4+C1+C2+C3":
    feature_based_combination( df_train, df_test, c = 10, x1 = 1, y1 = 4, x2 = 1, y2 = 3 )

      
