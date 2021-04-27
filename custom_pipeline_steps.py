import numpy as np

from imblearn import FunctionSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler

from sklearn.base import BaseEstimator,TransformerMixin


class SourceCombiner(BaseEstimator,TransformerMixin):
    def __init__(self,win=5):
        self.win=win

    def fit(self, X, y=None):
        return self

    def transform(self, X):
       out=[]
       for ii in X:
          t0=int(ii.shape[1]/2)
          out.append(np.mean(ii[:,t0-self.win:t0+self.win+1],1))
          #out.append(np.mean(ii[:,t0-self.win:t0+1],1))
                   
       return out


class CustomOverSampler(FunctionSampler):

   def __init__(self,random_state=11901, k_neighbors=0, accept_sparse=True, kw_args=None):

      super(FunctionSampler,self).__init__()

      def SMOTE_fun(X,y):
         return SMOTE(random_state=random_state,k_neighbors=k_neighbors).fit_resample(X,y)

      def RandomOverSampler_fun(X,y):
         return RandomOverSampler(random_state=random_state).fit_resample(X,y)

      if k_neighbors > 0:
         self.func=SMOTE_fun
      else:
         self.func=RandomOverSampler_fun

      self.k_neighbors=k_neighbors
      self.accept_sparse=accept_sparse
      self.kw_args=kw_args

