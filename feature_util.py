'''
Contains methods to extract features for training
'''
import pandas as pd

'''
Feature X vector (a column) to be added to final features data frame
'''
def get_feature_x(train):
  feature = []
  #ensure the size is as required
  return feature

'''
Returns features from all the feature extraction methods
'''
def get_features(train):
  features = pd.DataFrame(columns = colNames)
  feature['X'] = get_feature_x(train)
  return features
