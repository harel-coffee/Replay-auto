import numpy as np
from scipy.signal import resample
from scipy.stats import percentileofscore
from scipy.spatial.distance import hamming
import sys, pdb, mne, itertools, re, argparse, os, dill
import sklearn.externals.joblib as joblib
import multiprocessing as mp
import matplotlib.pyplot as plt

def get_itr_data(b0,bf,subj,day):

   X_itr=[]
   for bb in range(b0,bf+1): 

      try:
         # data of the entire bb-th rest trial
         print 'kp_source_broad/%s.%s.%.2d.rest.X.npy'%(subj,day,bb)
         X_itr.append( np.load('kp_source_broad/%s.%s.%.2d.rest.X.npy'%(subj,day,bb)) )
         
      except:
         print 'Problem loading itr data from %s block %d'%(subj,bb)
         continue

   if len(X_itr) > 0:
      return np.vstack(X_itr)
   else:
      return np.empty(0,)

def get_rest_data(subj,day):

   Xr_file='kp_source_broad/%s.%s.X.npy'%(subj,day)
   print Xr_file

   if os.path.exists(Xr_file):
      return(np.load(Xr_file))
   else:
      return []


def log_all_seqs(out):

   p,win,ind=out
   p_all_seqs[ind]=p
   all_wins[ind]=win

   print 'Logged seq ind %d'%ind


def calc_full_prob(X, prob, ind_seq, seq, ts, block):

   #p_seq=np.empty((prob.shape[0],))
   #p_seq[:]=np.nan

   all_seqs=list(itertools.product([1,2,3,4],repeat=5))
   p_seq=np.zeros((prob.shape[0],))

   wins=[[] for ii in enumerate(prob)]
   for ind, ii in enumerate(prob):
         if ind%2000 < 5*ts-1 and block == 'itr':
            continue
         elif ind >= 5*ts-1:
            win=prob[ind-5*ts+1:ind+1]
            if ts > 1:
               Xs = resample(X[ind-5*ts+1:ind+1],5,window=None)
               win=best_estimator.predict_proba(Xs)

            # Find the equivalence set of seq
            #eq_seq=[pp for pp in all_seqs if hamming(pp,seq) <= 0.2]
            eq_seq=[seq]
            p_eq_seq=[]
            for kk in eq_seq:

               p0 = win[0][kk[0]-1]
               p1 = win[1][kk[1]-1]
               p2 = win[2][kk[2]-1]
               p3 = win[3][kk[3]-1]
               p4 = win[4][kk[4]-1]

               p_eq_seq.append(p0*p1*p2*p3*p4)
            
            p_seq[ind] = np.max(p_eq_seq)

            wins[ind]=win 

   return p_seq,wins,ind_seq
   

def seq_score(X, prob, ts, ind_seq, thr):

   pvalue=np.empty((prob.shape[0],))
   pvalue[:]=1
   win_target=[]
   
   perms=list(itertools.product([1,2,3,4],repeat=5))

   for ind, ii in enumerate(prob):

      if p_all_seqs[ind_seq][ind] > thr:

         p_seq_null=np.zeros((len(perms),))

         for ind_jj,jj in enumerate(perms):

            p0 =all_wins[ind_seq][ind][0][jj[0]-1]
            p1 =all_wins[ind_seq][ind][1][jj[1]-1]
            p2 =all_wins[ind_seq][ind][2][jj[2]-1]
            p3 =all_wins[ind_seq][ind][3][jj[3]-1]
            p4 =all_wins[ind_seq][ind][4][jj[4]-1]
            p_seq_null[ind_jj] = p0*p1*p2*p3*p4
        
         pvalue[ind]=(100-percentileofscore(p_seq_null, p_all_seqs[ind_seq][ind],'strict'))/100.0
         ind_sort=np.argsort(p_seq_null)

         if pvalue[ind] < 0.05:
            win_target.append(all_wins[ind_seq][ind])

   if len(win_target) == 0:
      mean_win=np.empty((5,prob.shape[1]))
      mean_win[:]=np.nan
   else:
      mean_win=np.mean(win_target,0)

   return pvalue, mean_win

       
##########################################################
# Helper parallel functions
##########################################################
def compute(Xr, prob_Xr, ind, seq, ts, whichtests, whichcounts, whichrepl, whichwin, whichpval, thr):

   tests = np.sum(p_all_seqs[ind] > thr)

   pval_unc,win= seq_score(Xr, prob_Xr, ts, ind, thr)

   reject = pval_unc < 0.05
   counts=np.sum(reject)

   return tests, counts, whichtests, whichcounts, whichrepl, whichpval,whichwin,ind,\
          seq, thr, reject, pval_unc,win


def log_result(out):

   tests, counts, whichtests, whichcounts, whichrepl, whichpval, whichwin, ind, seq, thr,\
   reject,pval,win=out

   globals()[whichtests][ind]=tests
   globals()[whichcounts][ind]=counts
   globals()[whichrepl][ind]=reject
   globals()[whichpval][ind]=pval
   globals()[whichwin][ind]=win
   
   print 'Done %s[%d=%s]=%d/%d,(exp FP=%.2f|alpha=0.05) thr=%.2f'\
   %(whichcounts,ind,str(seq),counts, tests, 0.01*tests, thr)

   
if __name__=='__main__':

   # Find all subject codes from meg.txt file
   all_subjs=re.findall('([A-Z]+)\.day1\/.+',open('meg.txt').read())

   # Parse input arguments
   parser = argparse.ArgumentParser(description='Test key press decoding')
   parser.add_argument('--meg_codes',dest='subjs',default=all_subjs,nargs='+')
   parser.add_argument('--pickseqs',dest='pickseqs',default='',type=str)
   parser.add_argument('--b0',dest='b0',type=int)
   parser.add_argument('--bf',dest='bf',type=int)
   parser.add_argument('--dir',dest='dir',type=str,default='seq_decoder_broad')
   parser.add_argument('--njobs',dest='njobs',type=int)
   parser.add_argument('--compr',dest='compr',type=str)
   parser.add_argument('--dayblock',dest='dayblock',type=str)
   parser.add_argument('--trainday',dest='trainday',type=str,default='')
   args=parser.parse_args()

   dict_compr={'0.5'  : 160, 
               '1'    : 80,
               '1.25' : 64,
               '2.67' : 30,
               '4'    : 20,
               '5.33' : 15,
               '8'    : 10,
               '8.89' : 9,
               '10'   : 8,
               '11.43': 7,
               '13.33': 6,
               '16'   : 5,
               '20'   : 4,
               '26.67': 3,
               '40'   : 2,
               '80'   : 1}

   for ss in args.subjs:

      print ss

      allperms=list(itertools.product([1,2,3,4],repeat=5))
      postfix=''
      
      if args.pickseqs == 'seqs.txt': # read sequences from file
         
         perms  = [tuple(int(jj) for jj in ii) for ii in re.findall('(\d+)',open('seqs.txt','rt').read())]

      elif len(args.pickseqs) > 0: # read string with range of sequences
         
         rng = re.findall('(\d+)_(\d+)',args.pickseqs)[0]
         perms = allperms[int(rng[0]):int(rng[1])+1]

         postfix=args.pickseqs

      elif len(args.pickseqs) == 0: # all sequences

         perms = allperms

      filename='/scratch/claudinolm/%s/%s.%d.%d.%sx.%s.%s.pkl'\
      %(args.dir,ss,args.b0,args.bf,args.compr,args.dayblock,postfix)

      if os.path.exists(filename):
         continue

      ##########################################################
      # Reading data and key press model
      ##########################################################
      try:

         dayblock = re.findall('(day\d[rand]*)',args.dayblock)[0]

         if args.trainday == '':
            args.trainday=dayblock

         pickle_dict={ ii[2]: (ii[3],float(ii[0]),float(ii[1]))\
         for ii in re.findall('(\d+\.\d+),\s(\d+\.\d+),\s([A-Z]+),\s(.+)',\
         open('kp_decoder_broad/%s_models.txt'%args.trainday).read())}

         kp_filename=pickle_dict[ss][0]
         rate=pickle_dict[ss][1]
         #thr=pickle_dict[ss][2]

         model= dill.load(open(kp_filename))

         #best_estimator=model[0].steps[3][1]
         best_estimator=model[0]
         best_estimator.steps.pop(0)
         cv_conf_mat=model[-4]

         if args.dayblock.find('itr') >=0:
            print 'ITR'
            Xr=get_itr_data(args.b0,args.bf,ss,dayblock)
            block='itr'

         else:
            print 'Not ITR'
            Xr=get_rest_data(ss,args.dayblock)
            block='rest'

      except:
         print pickle_dict[ss]
         continue

      if len(Xr) == 0:
         continue

      ##########################################################
      # Find threshold from training data
      ##########################################################

      p_max_fp=[]
      for ind_ii,ii in enumerate(cv_conf_mat):
         p_max_fp.append(np.max(ii[range(0,ind_ii)+\
         range(ind_ii+1,len(cv_conf_mat))])/np.sum(ii).astype('float'))

      p_max_fp = np.prod(np.array(p_max_fp))
      thr = p_max_fp

      ##########################################################
      # Sequence processing
      ##########################################################
    

      p_c_Xr=best_estimator.predict_proba(Xr)


      tests_Xr = np.zeros((len(perms),))
      counts_Xr = np.zeros((len(perms),))
      mean_prob_Xr = np.zeros((len(perms),))
      replay = np.zeros((len(perms),Xr.shape[0]))
      pval = np.ones((len(perms),Xr.shape[0]))

      # Calculate probability of each sequence during the given dayblock
      all_wins= [ [] for ii in perms ]
      win = [ [] for ii in perms ]
      p_all_seqs=np.empty((len(perms),p_c_Xr.shape[0]))
      pool = mp.Pool(processes=args.njobs)
      for ind, seq in enumerate(perms):
         print ind,seq
         pool.apply_async(calc_full_prob,(Xr, p_c_Xr, ind, seq, dict_compr[args.compr],block),
         callback=log_all_seqs)
         #log_all_seqs(calc_full_prob(Xr, p_c_Xr, ind, seq, dict_compr[args.compr],block))
      pool.close()
      pool.join()

      pool = mp.Pool(processes=args.njobs)
      for ind, seq in enumerate(perms):
         
         print seq

         print thr

         pool.apply_async(compute,(Xr,p_c_Xr, ind, 
         seq,dict_compr[args.compr],'tests_Xr','counts_Xr',
         'replay', 'win','pval',thr), callback=log_result)

         #log_result(compute(Xr,p_c_Xr, ind, 
         #seq,dict_compr[args.compr],'tests_Xr','counts_Xr',
         #'replay', 'win', 'pval',thr))


      pool.close()
      pool.join()

      joblib.dump((tests_Xr,
                   counts_Xr,
                   mean_prob_Xr,
                   replay, pval, win, thr, perms),filename)
