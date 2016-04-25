import re
import pandas as pd
import numpy as np

from gensim import corpora, models, similarities
from difflib import SequenceMatcher

from build_tfidf import split


def ratio(w1, w2):
  '''
  Calculate the matching ratio between 2 words.
  Only account for word pairs with at least 90% similarity
  '''
  m = SequenceMatcher(None, w1, w2)
  r = m.ratio()
  if r < 0.9: r = 0.0
  return r


def build_features(data, tfidf, dictionary):
  '''
  Generate features:
    1. Cosine similarity between tf-idf vectors of query vs. title
    2. Cosine similarity between tf-idf vectors of query vs. description
    3. Cosine similarity between tf-idf vectors of query vs. attribute text
    4. Sum of word match ratios between query vs. title
    5. Sum of word match ratios between query vs. description
    6. Sum of word match ratios between query vs. attribute text
    7. Query word count
  '''
  result = []
  for loc in xrange(len(data)):
    rowdata = data.loc[loc, ["product_title", "product_description", "attr_value", "search_term"]]
    rowbow = [[str(text)] if isinstance(text, float) else split(text) for text in rowdata]
    
    # query match level
    titleMatch = descMatch = attrMatch = 0
    for q in rowbow[3]:
      titleMatch = titleMatch + np.sum(map(lambda w: ratio(q, w), rowbow[0]))
      descMatch = descMatch + np.sum(map(lambda w: ratio(q, w), rowbow[1]))
      attrMatch = attrMatch + np.sum(map(lambda w: ratio(q, w), rowbow[2]))

    # get tfidf vectors
    rowdata = [tfidf[dictionary.doc2bow(text)] for text in rowbow]

    # prepare to get similarities
    index = similarities.SparseMatrixSimilarity(rowdata[:3], num_features=len(dictionary))

    # append everything to the result
    result.append(np.concatenate((index[rowdata[3]], [titleMatch, descMatch, attrMatch, len(rowbow[3])]), axis=0).tolist())
  # end loop
  return np.array(result)


def main():
  # load data
  df_desc = pd.read_csv('data/product_descriptions.csv', encoding="ISO-8859-1")
  df_attr = pd.read_csv('data/attributes_combined.csv', encoding="ISO-8859-1")

  df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
  df_train = pd.merge(df_train, df_desc, how='left', on='product_uid')
  df_train = pd.merge(df_train, df_attr, how='left', on='product_uid')

  df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
  df_test = pd.merge(df_test, df_desc, how='left', on='product_uid')
  df_test = pd.merge(df_test, df_attr, how='left', on='product_uid')

  # load tfidf model
  dictionary = corpora.Dictionary.load('homedepot.dict')
  corpus = corpora.MmCorpus('homedepot.mm')
  tfidf = models.TfidfModel.load('homedepot.tfidf')
  
  # build features
  trainData = build_features(df_train, tfidf, dictionary)
  testData = build_features(df_test, tfidf, dictionary)
  
  # save to csv
  df = pd.DataFrame(trainData, columns=['qt', 'qd', 'qa', 'mt', 'md', 'ma', 'ql'])
  df.to_csv('data/train_features.csv', index=False)
  df = pd.DataFrame(testData, columns=['qt', 'qd', 'qa', 'mt', 'md', 'ma', 'ql'])
  df.to_csv('data/test_features.csv', index=False)
  

if __name__ == "__main__":
	main()