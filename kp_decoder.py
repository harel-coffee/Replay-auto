import numpy as np, pdb, argparse, re, itertools, dill
from scipy.io import savemat

from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import StratifiedKFold,cross_val_predict,\
permutation_test_score, GridSearchCV
from sklearn.preprocessing import FunctionTransformer

from sklearn.metrics import confusion_matrix,f1_score
from sklearn.preprocessing import RobustScaler
import sklearn.svm as svm
import sklearn.externals.joblib as joblib

# Need to import from outside module so pickling/unpickling works
from custom_score_functions import proba_pred_accuracy, proba_pred_bal_accuracy, proba_pred_deltaprob_accuracy,\
proba_pred_weighted_accuracy
from custom_pipeline_steps import SourceCombiner, CustomOverSampler

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline


def get_data_to_fit(b0,bf,subj,dayblock):

   X_train=[]
   y_train=[]

   for bb in range(b0,bf+1): 

      # Load bb-th practice trial and corresponding key labels
      try:
         # Windows centered at key presses in bb-th trial
         X_e=np.load('kp_source_broad/%s.%s.%.2d.motor.epochs.X.npy'%(subj,dayblock,bb)) 
         
         # Key index of each X_e and X
         y=np.load('kp_source_broad/%s.%s.%.2d.motor.epochs.y.npy'%(subj,dayblock,bb))
      
      except:
         print 'Problem loading data from  %s block %d'%(subj,bb)
         continue

      # Will use center sample to train
      X_train.append(X_e)
      y_train.append(y)

   y_train=np.hstack(y_train)

   return X_train, y_train


if __name__=='__main__':

   # Find all subject codes from meg.txt file
   all_subjs=re.findall('([A-Z]+)\.day1\/.+',open('meg.txt').read())

   # Parse input arguments
   parser = argparse.ArgumentParser(description='Test key press decoding')
   parser.add_argument('--meg_codes',dest='subjs',default=all_subjs,nargs='+')
   parser.add_argument('--dayblock',dest='dayblock',type=str)
   parser.add_argument('--crit',dest='crit_fun',type=str)
   parser.add_argument('--folds',dest='folds',type=int)
   parser.add_argument('--b0',dest='b0',type=int)
   parser.add_argument('--bf',dest='bf',type=int)
   parser.add_argument('--njobs',dest='njobs',type=int)
   args=parser.parse_args()

   if args.crit_fun == 'proba_pred_bal_accuracy':
      print 'Using custom criterion fun proba_pred_bal_accuracy'
      crit_fun=proba_pred_bal_accuracy
   elif args.crit_fun == 'proba_pred_accuracy':
      print 'Using custom criterion fun proba_pred_accuracy'
      crit_fun=proba_pred_accuracy
   elif args.crit_fun == 'proba_pred_weighted_accuracy':
      print 'Using custom criterion fun proba_pred_weighted_accuracy'
      crit_fun=proba_pred_weighted_accuracy
   elif args.crit_fun == 'proba_pred_deltaprob_accuracy':
      print 'Using custom criterion fun proba_pred_deltaprob_accuracy'
      crit_fun=proba_pred_deltaprob_accuracy
   else:
      print 'Using sklearn criterion fun f1_macro'
      crit_fun=args.crit_fun

   for ind_ss, ss in enumerate(args.subjs):

      ##########################################################
      # Fitting key press model
      ##########################################################
      # Reads data
      X_train,y_train = get_data_to_fit(args.b0,args.bf,ss,args.dayblock)
      y_train_str=''.join([str(ii) for ii in y_train])
      X_train=list(itertools.chain(*X_train))

      start_seq=np.array([ii.start() for ii in re.finditer('41324',y_train_str)])
      inds_seq=list(itertools.chain(*[range(z[0],z[1]) for z in [(ii,jj) for ii,jj in zip(start_seq,start_seq+5)]]))
      
      # Will keep only keypresses of full correct sequences
      X_train = [X_train[ii] for ii in inds_seq]
      y_train=y_train[inds_seq]

      inds=np.random.RandomState(seed=11901).permutation(range(len(X_train)))

      X_train = [X_train[ii] for ind,ii in enumerate(inds)]
      y_train = y_train[inds]

      pdb.set_trace()

      grid_sss=StratifiedKFold(n_splits=args.folds, shuffle=False, random_state=11901)
      split=list(grid_sss.split(X_train,y_train))

      #C=[0.05,0.1,0.25,0.5,0.75,1,5,10,25,50,75,100,1000]
      #gamma=[0.05,0.075,(1-args.folds/10.0)/len(X_train),0.1,0.5,0.75,1]
      ##k_nn=[0,1,5]
      #scale=[True,False]
      #win=[5,10,25,50,75,100,125,150,175,200]

      C=[0.1,0.5,1,10,50,100,1000]
      gamma=[0.05,0.1,(1-args.folds/10.0)/len(X_train),0.1,0.5,1]
      scale=[True,False]
      win=[5,10,25]

      # To quickly check stuff
      #C=[1,100]
      #gamma=[1.0/len(X_train),1]
      #scale=[True,False]
      #win=[5,10]

      params = {'svc__gamma':gamma, 'svc__C':C, 'feature__win':win,
                'scaler__with_scaling':scale,
                'scaler__with_centering':scale}

      #'sampler__k_neighbors':k_nn,


      estimator= Pipeline(steps=[
      ('feature',SourceCombiner()),
      ('sampler',RandomOverSampler(random_state=11901)),
      ('scaler',RobustScaler()),
      ('svc', svm.SVC(kernel='rbf',shrinking=False, probability=True,
      tol=0.001,cache_size=20000, class_weight=None, verbose=False, 
      max_iter=-1, decision_function_shape='ovr', random_state=11901))])

      print 'Grid searching hyperparams'

      grid = GridSearchCV(estimator, cv=split, n_jobs=-1, param_grid=params, 
      scoring=crit_fun,refit=True, verbose=2)

      grid.fit(X_train,y_train)
      
      ##########################################################
      # Running permutations on key press model
      ##########################################################

      # Note: running cross_val score with the same split and scoring function will match the 
      # permutation_test_score:
      #
      # [score, perm_score, p] = permutation_test_score(grid.best_estimator_, 
      # X_train, y_train, random_state=11901, scoring=crit_fun, cv=split, 
      # n_permutations=1000, n_jobs=args.njobs)
      #
      # cv_score = cross_val_score(grid.best_estimator_,X_train,y_train,cv=split,scoring=proba_pred_bal_accuracy)
      #
      # results in: cv_score = np.mean(score)


      print 'Running permutations'

      # Check if best configuration works using the same test splits where
      # predictions were calculated

      [score, perm_score, p] = permutation_test_score(grid.best_estimator_, 
      X_train, y_train, random_state=11901, scoring=crit_fun, cv=split, 
      n_permutations=1000, n_jobs=args.njobs)

      print 'Across-folds average score of best configuration out of GridSearchCV=%.4f'%grid.best_score_

      ##########################################################
      # Key press decoder diagnosis
      ##########################################################

      print 'True score using permutation_test_score()=%.4f'%score
      print 'Max null score=%.4f'%np.max(perm_score)
      print 'P-value: %.4f'%p

      ##########################################################
      # Cross val confusion matrix
      ##########################################################

      # Note: running cross_val score with the same split and scoring function will match the cv_results 
      # outputs of the grid, with a little roundoff error, that is after running:
      #
      # score=cross_val_score(grid.best_estimator_,X_train,y_train,cv=split,scoring=proba_pred_bal_accuracy)
      #
      # grid.cv_results_['split0_test_score'][grid.best_index_] = score[0] 
      # grid.cv_results_['split1_test_score'][grid.best_index_] = score[1]
      # ... 
      # grid.cv_results_['split4_test_score'][grid.best_index_] = score[4] 
      #
      # and also
      # grid.cv_results_.best_score_ = np.mean(score)

      # Cross-validated estimations
      pred_probs=cross_val_predict(grid.best_estimator_,X_train,y_train,cv=split,method='predict_proba')
      y_pred=np.array([np.argmax(ii)+1 for ii in pred_probs])
      
      # Cross-validated confusion matrix
      cv_conf_mat=confusion_matrix(y_train,y_pred)
      savemat('kp_decoder_broad/%s.cv_cnf_mat.%s.%s'%(ss,args.dayblock,args.crit_fun),{'cv_cnf_mat':cv_conf_mat})

      print 'Cross-validated confusion matrix'
      print cv_conf_mat

      y_pred=np.array([np.argmax(ii)+1 for ii in grid.best_estimator_.predict_proba(X_train)])
      train_conf_mat=confusion_matrix(y_train,y_pred)

      print 'Confusion matrix calculated on full training data'
      print train_conf_mat

      # Re-train best configuration on entire data
      grid.best_estimator_.fit(X_train,y_train)

      yt=np.zeros((len(y_train),))
      pp = np.zeros(pred_probs.shape)
      for ind,ii in enumerate(inds):
         yt[ii]=y_train[ind]
         pp[ii]=pred_probs[ind]

      pred_probs=[pp[jj,:] for jj in [range(ii,ii+5) for ii in range(len(pp))[::5]]]
 
      with open('kp_decoder_broad/%s.cv.%s.%s.pkl'%(ss,args.dayblock,args.crit_fun),'wb') as file:
         dill.dump((grid.best_estimator_, grid.best_score_, grid.best_params_,
                   score, perm_score, p, cv_conf_mat, train_conf_mat, inds_seq, pred_probs),file)

      '''
      pdb.set_trace()

      sample=[]
      for sp in range(1000):
         z=[np.random.choice(size=1,a=[1,2,3,4],p=kk)[0] for kk in probs[jj]\
         for jj in [range(ii,ii+5) for ii in range(len(probs))[::5]]]
         sample.append(np.matrix(z).reshape(len(z)/5,5))

      print len(sample)
      '''
