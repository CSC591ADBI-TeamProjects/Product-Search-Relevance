import pandas as pd
import numpy as np
import re
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS
from sklearn import ensemble
from sklearn.cross_validation import train_test_split

def split(text):
  delimiters = ".", ",", ";", ":", "-", "(", ")", " ", "\t"
  regexPattern = '|'.join(map(re.escape, delimiters))
  return [word for word in re.split(regexPattern, text.lower()) if word not in STOPWORDS and word != ""]

def main():
  df_train = pd.read_csv('../../data/train.csv', encoding="ISO-8859-1")
  df_desc = pd.read_csv('../../data/product_descriptions.csv', encoding="ISO-8859-1")
  df_attr = pd.read_csv('../../data/attributes_combined.csv', encoding="ISO-8859-1")
  
  # split the texts
  titles = [split(line) for line in df_train["product_title"]]
  descs = [split(line) for line in df_desc["product_description"]]
  attrs = [[str(line)] if isinstance(line, float) else split(line) for line in df_attr["attr_value"]]
  queries = [split(line) for line in df_train["search_term"]]
  texts = np.concatenate((titles, descs, attrs, queries))

  # remove infrequent words
  from collections import defaultdict
  frequency = defaultdict(int)
  for text in texts:
    for token in text:
      frequency[token] += 1
  texts = [[token for token in text if frequency[token] > 2] for text in texts]

  # build dictionary
  dictionary = corpora.Dictionary(texts)
  dictionary.save('homedepot.dict')
  print dictionary

  # actually build a bag-of-words corpus
  corpus = [dictionary.doc2bow(text) for text in texts]
  corpora.MmCorpus.serialize('homedepot.mm', corpus)

  # build Tf-idf model
  tfidf = models.TfidfModel(corpus)
  tfidf.save('homedepot.tfidf')


if __name__ == "__main__":
	main()