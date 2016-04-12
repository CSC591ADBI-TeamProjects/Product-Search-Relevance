'''
Contains methods to extract features for training
'''
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
'''
Feature X vector (a column) to be added to final features data frame
'''
def get_feature_x(train):
  feature = [] 
  return feature
  
'''
Returns 2 feature lists with count of words in 
search term that occur title and description resp.
'''
def get_feature_word_match(train):
  feature1 = [] 
  feature2 = []
  for _ , row in train.iterrows():
    searchWordList = re.sub("[^\w]", " ", row["search_term"]).split()
    titleWordList = re.sub("[^\w]", " ",  row["product_title"]).split()
    descWordList = re.sub("[^\w]", " ",  row["product_description"]).split()
    for word in searchWordList:
      feature1.append(titleWordList.count(word))
      feature2.append(descWordList.count(word))
      
  return (feature1, feature2)

'''
Returns feature: search query length
'''
def get_feature_query_length(train):
  feature = []
  for query in train["search_term"]:
  	feature.append(len(query.split())) 
  	
  return [float(x)/sum(feature) for x in feature]
  
'''
Returns feature: Tf-Idf
'''
def get_feature_Tf_Idf(train):
  feature1 = []
  feature2 = []
  #ensure the size is as required
  corpus_title = [x for x in train["product_title"]]
  corpus_description = [x for x in train["product_description"]]
  vectorizer1 = TfidfVectorizer(max_df = 1)
  vectorizer2 = TfidfVectorizer(max_df = 1)
  X1 = vectorizer1.fit_transform(corpus_title)
  X2 = vectorizer2.fit_transform(corpus_description)
  idf1 = vectorizer1.idf_
  idf2 = vectorizer2.idf_
  feature1.append(idf1)
  feature2.append(idf2)
  return (feature1, feature2)
  


'''
Returns feature: Cosine Similarity
'''
def get_feature_cosine_similarity(train):
  feature_prod_title = []
  feature_prod_desc = []
  #ensure the size is as required
  vect = TfidfVectorizer(min_df=1)
  for _ , row in train.iterrows():
    cos_prod_title = vect.fit_transform([row["product_title"],row["search_term"]])
    cos_prod_desc = vect.fit_transform([row["product_description"],row["search_term"]])
    feature_prod_title.append((cos_prod_title*cos_prod_title.T).A[0][1])
    feature_prod_desc.append((cos_prod_desc*cos_prod_desc.T).A[0][1])
  return feature_prod_title,feature_prod_desc

'''
Returns features from all the feature extraction methods
'''
def get_features(train):
  colNames = []
  features = pd.DataFrame(columns = colNames)
  features['title_match_count'], features['desc_match_count'] = get_feature_word_match(train)
  
  return features

