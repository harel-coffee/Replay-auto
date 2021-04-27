import numpy as np

def proba_pred_accuracy(estimator, X, y):

   y_pred=[np.argmax(ii)+1 for ii in estimator.predict_proba(X)]

   return np.sum(y == y_pred)/float(len(y))


def proba_pred_bal_accuracy(estimator, X, y):

   y_pred=np.array([np.argmax(ii)+1 for ii in estimator.predict_proba(X)])

   acc=[]                           
   for ii in np.unique(y):
      inds_ii = y==ii
      acc.append(np.sum(y_pred[inds_ii]==y[inds_ii])/float(np.sum(inds_ii)))

   return np.mean(acc)

def proba_pred_weighted_accuracy(estimator, X, y):

   y_pred=np.array([np.argmax(ii)+1 for ii in estimator.predict_proba(X)])

   weight=np.array([2.0/7,2.0/7,2.0/7,1.0/7])

   acc=[]                           
   for ii in np.unique(y):
      inds_ii = y==ii
      acc.append(np.sum(y_pred[inds_ii]==y[inds_ii])/float(np.sum(inds_ii)))

   return weight[np.unique(y)-1].dot(np.array(acc))


def proba_pred_deltaprob_accuracy(estimator, X, y):

   pred_probs=estimator.predict_proba(X)
   acc=[]
   for ii in np.unique(y):
      
      probs_ii = pred_probs[np.where(y==ii)]
      not_ii=np.arange(len(np.unique(y)))<>ii-1      

      acc.append( np.mean(probs_ii[:,ii-1]-np.max(probs_ii[:,not_ii] ,1)) )

   return np.mean(acc)
