import sys
import data_util as dutil
import pandas as pd
import model_util
import feature_util

if len(sys.argv) != 3:
    print "python main.py <path_to_data_directory> <model_name>"
    print "models: glm, ..."
    exit(1)

model_param = sys.argv[2]
path_to_data = sys.argv[1]

def main():
#  df_all_train,df_test = dutil.join_data(path_to_data)
  features = feature_util.get_features(df_all_train)
  result = getattr(model_util, model_param)(df_all_train,df_test)


if __name__=="__main__":
  main()
