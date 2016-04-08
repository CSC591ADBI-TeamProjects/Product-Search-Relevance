import sys
import pandas as pd
from nltk.stem.snowball import SnowballStemmer

#path_to_data = "data/"
stemmer = SnowballStemmer("english")

def join_data(path_to_data):
  df_train = pd.read_csv(path_to_data+'train.csv', encoding="ISO-8859-1")
  #stemming of training data
  df_train["search_term"] = df_train['search_term'].map(lambda x:str_stemmer(x))
  df_train['product_title'] = df_train['product_title'].map(lambda x:str_stemmer(x))
  
  df_test = pd.read_csv(path_to_data+'test.csv', encoding="ISO-8859-1")
  
  df_pro_desc = pd.read_csv(path_to_data+'product_descriptions.csv', encoding="ISO-8859-1")
  #stemming of product_description data - takes ENORMOUS TIME, remove if not needed
  df_pro_desc['product_description'] = df_pro_desc['product_description'].map(lambda x:str_stemmer(x))
  
  #TODO do stemming for attributes data also
  df_attr = pd.read_csv(path_to_data+'attributes.csv', encoding="ISO-8859-1")
  
  df_desc_attr = pd.merge(df_pro_desc, df_attr, how='left', on='product_uid')
  df_all = pd.merge(df_train, df_desc_attr, how='left', on='product_uid')
  return df_all,df_test

#takes a string and returns a string with words replaced with their stems
def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])
