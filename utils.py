import pandas as pd

#from sklearn.decomposition import NMF
import pickle 
import warnings
warnings.filterwarnings(action='ignore')

with open('./factorizer_NMF.pkl','rb') as file_in:
              model = pickle.load(file_in)
    # database
user_item=pd.read_csv('./user_item.csv', index_col=0)
MOVIES=user_item.columns.to_list()
USERS=user_item.index.to_list()
col_mean= user_item.mean()

