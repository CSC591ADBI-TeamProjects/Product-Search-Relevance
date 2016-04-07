import sys
import pandas as pd

path_to_data = "data/"

def join_data():
  #train = open(path_to_data+'train.csv','r')
  df_train = pd.read_csv(path_to_data+'train.csv', encoding="ISO-8859-1")
  df_test = pd.read_csv(path_to_data+'test.csv', encoding="ISO-8859-1")
  df_pro_desc = pd.read_csv(path_to_data+'product_descriptions.csv', encoding="ISO-8859-1")
  df_attr = pd.read_csv(path_to_data+'attributes.csv', encoding="ISO-8859-1")
  df_desc_attr = pd.merge(df_pro_desc, df_attr, how='left', on='product_uid')
  df_all = pd.merge(df_train, df_desc_attr, how='left', on='product_uid')
  return df_all