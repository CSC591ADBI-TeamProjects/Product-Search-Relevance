import re
import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import neural_network
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm

from sklearn.cross_validation import train_test_split

def main():
  df_train = pd.read_csv('../../data/train.csv', encoding="ISO-8859-1")
  trainData = pd.read_csv('../../data/train_features.csv', encoding="ISO-8859-1")
  X_train, X_test, y_train, y_test = train_test_split(trainData, df_train['relevance'], test_size=0.2, random_state=42)
  
  # Classifiers
  #model = ensemble.RandomForestClassifier(n_estimators=50, criterion='entropy')
  #model = ensemble.GradientBoostingClassifier()
  #model.fit(X_train, [str(n) for n in y_train])
  #print("Train RMSE: %.3f" % np.sqrt(np.mean(([float(n) for n in model.predict(X_test)] - y_test) ** 2)))
  
  # Regressors
  #model = linear_model.SGDRegressor()
  #model = ensemble.RandomForestRegressor()
  #model = ensemble.GradientBoostingRegressor()
  model = svm.LinearSVR()
  model.fit(X_train, y_train)
  print("Train RMSE: %.3f" % np.sqrt(np.mean((model.predict(X_test) - y_test) ** 2)))

  # Predict test data
  # df_test = pd.read_csv('../../data/test.csv', encoding="ISO-8859-1")
  # testData = pd.read_csv('../../data/test_features.csv', encoding="ISO-8859-1")
  # model = ensemble.GradientBoostingRegressor()
  # model.fit(trainData, df_train["relevance"])
  # prediction = model.predict(testData)
  # prediction = np.array([1 if arr < 1 else 3 if arr > 3 else arr for arr in prediction])
  # df = pd.DataFrame({'id': df_test['id'], 'relevance': pd.Series(prediction, dtype=float)})
  # df.to_csv('../../data/testpredict.csv', index=False)


if __name__ == "__main__":
	main()