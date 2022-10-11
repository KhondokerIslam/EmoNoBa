import os
import pandas as pd

def converting_to_cc(df):
  numerals = ['০','১','২','৩','৪','৫','৬','৭','৮','৯', '0','1','2','3','4','5','6','7','8','9']
  cnt = 0
  for sentence in df['Data']:
    st = ""
    sentence = sentence.split(' ')
    # print(sentence)
    for i in range(len(sentence)):
      
      if(len(sentence[i]) != 0 and sentence[i][0] in numerals):
        sentence[i] = "CC"
      st += sentence[i]
      st += " "
    
    df['Data'][cnt] = st
    cnt += 1

  return df

def remove_cons_puns(stri, temp):
  cnt = 0
  ha_cnt = 0
  for sentence in temp['Data']:
    sentence = sentence.split()
    st = ""
    for index in range(len(sentence)):
      if(sentence[index] == stri):
        if(index+1 < len(sentence) ):
          if(sentence[index+1] ==stri ):
            ha_cnt  += 1
            continue
          else:
            st += stri

      else:
        st += sentence[index]
      st += ' '
    temp['Data'][cnt] = st
    cnt = cnt + 1
      

    print(ha_cnt)
    return temp


df_train = pd.read_csv("../EmoNoBa Dataset/Train.csv")
df_train = remove_cons_puns(',', df_train)
df_train = remove_cons_puns('।', df_train)
df_train = converting_to_cc( df_train )

df_val = pd.read_csv("../EmoNoBa Dataset/Val.csv")
df_val = remove_cons_puns(',',df_val)
df_val = remove_cons_puns('।', df_val)
df_val = converting_to_cc( df_val )

df_test = pd.read_csv("../EmoNoBa Dataset/Test.csv")
df_test = remove_cons_puns(',',df_test)
df_test = remove_cons_puns('।', df_test)
df_test = converting_to_cc( df_test )

df_train.to_csv('../EmoNoBa Dataset/Final_Train.csv', index = False)
df_val.to_csv('../EmoNoBa Dataset/Final_Val.csv', index = False)
df_test.to_csv('../EmoNoBa Dataset/Final_Test.csv', index = False)