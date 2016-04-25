import re
import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import neural_network
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm

from sklearn.cross_validation import train_test_split

def roundup(num):
  '''
  Round regression numbers to fixed intervals of ground truth values: 1.00, 1.33, 1.67, 2.00, 2.33, 2.67, 3.00
  However, this doesn't help improve test score
  '''
  num = np.round(float(num) * 3) / 3
  if num < 1: num = 1
  if num > 3: num = 3
  return num


def main():
  '''
  Test different models with generated features
  '''
  df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
  trainData = pd.read_csv('data/train_features.csv', encoding="ISO-8859-1")
  X_train, X_test, y_train, y_test = train_test_split(trainData, df_train['relevance'], test_size=0.3, random_state=42)
  
  # Classifiers: weaker results than Regressors
  # model = ensemble.RandomForestClassifier(n_estimators=50, criterion='entropy')
  # model = ensemble.GradientBoostingClassifier()
  # model = svm.SVC()
  # model.fit(X_train, [str(n) for n in y_train])
  # print("Train RMSE: %.3f" % np.sqrt(np.mean(([float(n) for n in model.predict(X_test)] - y_test) ** 2)))
  
  # Regressors
  # model = linear_model.SGDRegressor()
  # model = ensemble.RandomForestRegressor()
  # model = svm.LinearSVR()
  # model = svm.NuSVR(kernel='poly')
  model = ensemble.GradientBoostingRegressor()
  model.fit(X_train, y_train)
  print("Train RMSE: %.3f" % np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))

  # Predict test data
  df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
  testData = pd.read_csv('data/test_features.csv', encoding="ISO-8859-1")
  model = ensemble.GradientBoostingRegressor()
  model.fit(trainData, df_train["relevance"])
  prediction = model.predict(testData)
  prediction = np.array([1 if arr < 1 else 3 if arr > 3 else arr for arr in prediction])
  # df = pd.DataFrame({'id': df_test['id'], 'relevance': pd.Series([roundup(num) for num in prediction], dtype=float)})
  df = pd.DataFrame({'id': df_test['id'], 'relevance': pd.Series(prediction, dtype=float)})
  df.to_csv('data/testpredict.csv', index=False, float_format='%.10f')


if __name__ == "__main__":
	main()