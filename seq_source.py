import matplotlib.pyplot as plt
import sklearn.neighbors as nn
import sklearn.linear_model as lm
import sklearn.svm as svm
import numpy as np
import sys, pdb, mne, itertools, re, argparse, os
import sklearn.externals.joblib as joblib
from scipy.stats import sem, ttest_rel, percentileofscore, pearsonr, spearmanr
from scipy.spatial.distance import hamming
import multiprocessing as mp
from statsmodels.stats.multitest import multipletests
from scipy.io import savemat



def get_source_data_rest(subj,dayblock):

   try:
      return np.load('kp_source_broad/%s.%s.X.npy'%(subj,dayblock))
   except:
      return np.empty(0,)  



def get_source_data_itr(b0,bf,subj,dayblock):

   Xr_iti=[]

   for bb in range(b0,bf+1):

      # Load bb-th practice trial and corresponding key labels
      try:
         # Data of the entire bb-th rest trial
         Xr=np.load('kp_source_broad/%s.%s.%.2d.rest.X.npy'%(subj,dayblock,bb))

      except:
         print 'Problem loading data from  %s block %d'%(subj,bb)
         continue

      Xr_iti.append(Xr)

   
   if len(Xr_iti) > 0:
      return np.vstack(Xr_iti)
   else:
      return np.empty(0,)



def get_source(X, replay, ts):

   Xs=[]
   for ind in range(X.shape[0]):
      if replay[ind] > 0:
         if ind-5*ts+1-5 >= 0 and ind+1+5 < X.shape[0]:
            Xs.append(X[ind-5*ts+1-5:ind+1+5,:]) # window -5*ts -5ms +5*ts +5ms samples

   if len(Xs) > 0:
      return np.nanmean(Xs,0)
   return Xs



def score_rest(ind_cc, cc, ind_ss, ss, n_seqs, dayblock, thr, filename_pre):

   if ss == 'KHHSEQKC':
      return [[]]*12

   try:

      pdb.set_trace()


      tests_Xr_pre,counts_Xr_pre,mean_prob_Xr_pre,\
      replay_pre,pval_unc_pre, win_pre, thr_pre,_=\
      joblib.load(filename_pre)

      # FDR correction
      replay_pre[:]=0
      for ind_ii,ii in enumerate(pval_unc_pre):
         pvals=ii[np.where(ii<1)]
         if len(pvals) > 0:
            reject,_,_,_ = multipletests(pvals,alpha=thr,method='fdr_bh')
            #reject = pvals<thr
            replay_pre[ind_ii,np.where(ii<1)] = reject


      pdb.set_trace()


     # Post process the replay raster to remove repeated detections
      for ind_ii,ii in enumerate(replay_pre):

         str_ii=''.join(str(ii.astype(int).tolist()).split(', '))
         coords=[(m.start(),m.end()-1) for m in re.finditer('1+',str_ii) if (m.end()-1-m.start())> 0]

         for jj in coords:
            replay_pre[ind_ii][jj[0]:jj[1]]=0

            replay_coord=np.round(sum(jj)/2)
            if replay_coord-1 > 0 and replay_pre[ind_ii][replay_coord-1] == 0:
               replay_pre[ind_ii][replay_coord]=1
               #print jj

      pdb.set_trace()

      # Apply lookback correction
      for ind_ii,ii in enumerate(replay_pre):
         coords = np.where(ii>0)[0]
         for jj in coords:
            coords_window = np.where(replay_pre[ind_ii][jj-lookback+1:jj+1]>0)[0]
            replay_pre[ind_ii][jj-lookback+1:jj+1]=0
            replay_pre[ind_ii][jj-lookback+int(np.median(coords_window))+1]=1


      pdb.set_trace()

      adj_pre=replay_pre.shape[1]/200.0

      Xr_pre=get_source_data_rest(ss, dayblock)
      src = get_source(Xr_pre, replay_pre[0], dict_compr[cc])

      pdb.set_trace()

      print 'Calculated source %s %s %s'%(dayblock,ss,cc)

      return np.sum(replay_pre,1)/adj_pre, probs_pre_per_s_ma, src, win_pre, ind_cc, ind_ss,-1,-1,\
      'probs_pre_per_s','probs_pre_per_s_ma', 'src_pre', 'win_seq_pre'


   except:

      print '>>Failed to process %s %s %s'%(dayblock,ss,cc)
      return [[]]*12



def score_itr(ind_cc, cc, ind_ss, ss, n_seqs, b0, bf, dayblock, thr, filename_iti):

   if ss == 'KHHSEQKC':
      return [[]]*12

   try:

      day=re.findall('(day\d)',dayblock)

      tests_Xr_iti,counts_Xr_iti,mean_prob_Xr_iti,\
      replay_iti,pval_unc_iti, win_iti, thr_iti,_=\
      joblib.load(filename_iti)

      # FDR correction
      replay_iti[:]=0
      for ind_ii,ii in enumerate(pval_unc_iti):
         pvals=ii[np.where(ii<1)]
         if len(pvals) > 0:
            reject,_,_,_ = multipletests(pvals,alpha=thr,method='fdr_bh')
            #reject = pvals<thr
            replay_iti[ind_ii,np.where(ii<1)] = reject


      # Post process the replay raster to remove repeated detections

      for ind_ii,ii in enumerate(replay_iti):

         str_ii=''.join(str(ii.astype(int).tolist()).split(', '))
         coords=[(m.start(),m.end()-1) for m in re.finditer('1+',str_ii) if (m.end()-1-m.start())> 0]

         for jj in coords:

            replay_iti[ind_ii][jj[0]:jj[1]]=0

            replay_coord=np.round(sum(jj)/2)
            if replay_coord-1 > 0 and replay_iti[ind_ii][replay_coord-1] == 0:
               replay_iti[ind_ii][replay_coord]=1
               #print jj

      # Apply lookback correction
      for ind_ii,ii in enumerate(replay_iti):
         coords = np.where(ii>0)[0]
         for jj in coords:
            coords_window = np.where(replay_iti[ind_ii][jj-lookback+1:jj+1]>0)[0]
            replay_iti[ind_ii][jj-lookback+1:jj+1]=0
            replay_iti[ind_ii][jj-lookback+int(np.median(coords_window))+1]=1


      adj_iti=replay_iti.shape[1]/200.0

      if os.path.exists('etc/%s.%s.valid_trials.txt'%(ss,day)): 
         trials=[int(ii) for ii in open('etc/%s.%s.valid_trials.txt'%(ss,day)).read().split(' ')]
         trial_nos=trials[0:5]
         trial_nos = (trial_nos[-1]-trial_nos[0]+1)*2000 - trial_nos[0]*2000
         ini_trial=np.maximum(trials[0],b0)
         fin_trial=np.minimum(trials[-1],bf)
      else:
         trial_nos=10000
         ini_trial=b0
         fin_trial=bf
         
      print ini_trial,fin_trial      

      p_seq_iti_per_s_per_trial=[]
      for jj in range(ini_trial,fin_trial+1):
         p_seq_iti_per_s_per_trial.append(np.sum(replay_iti[:,jj*2000:(jj+1)*2000-1],1)/(2000/200.0))

      print 'Calculated before source %s %s %s'%(dayblock,ss,cc)
      print ini_trial, fin_trial

      day=re.findall('(day\d)',dayblock)[0]
      Xr_iti=get_source_data_itr(ini_trial, fin_trial, ss, day)
      src = get_source(Xr_iti, replay_iti[0], dict_compr[cc])

      print 'Calculated source %s %s %s'%(dayblock,ss,cc)


      return np.sum(replay_iti,1)/adj_iti, p_seq_iti_per_s_per_trial, src, win_iti, ind_cc, ind_ss,\
         ini_trial, fin_trial,'probs_iti_per_s','probs_iti_per_s_per_trial', 'src_iti', 'win_seq_iti'

   except:
      print '>>Failed to process %s %s %s'%(dayblock,ss,cc)
      return [[]]*12


def log_result(out):

   #print 'Will log!'

   p_per_s, p_per_s_ma, src, win, ind_cc, ind_ss, ini, fin, which_per_s, which_ma, which_src, which_win = out

   if len(p_per_s) == 0:
      return

   globals()[which_win][ind_cc][ind_ss]=win

   globals()[which_per_s][ind_cc][ind_ss]=p_per_s

   globals()[which_src][ind_cc][ind_ss]=src

   if ini < 0:
      globals()[which_ma][ind_cc][ind_ss]=p_per_s_ma
   else:
      globals()[which_ma][ind_cc][ind_ss][ini:fin+1]=p_per_s_ma

   #print 'Logged!'


def control(subjs, njobs, dayblock, b0, bf, thr, n_seqs, indir, outdir):
   
   global probs_pre_per_s, probs_pre_per_s_ma, src_pre, win_seq_pre,\
          probs_iti_per_s, probs_iti_per_s_per_trial, src_iti, win_seq_iti, dict_compr

   compr=['0.5','1','1.25','2.67','4','5.33','8','8.89',\
   '10','11.43','13.33','16','20','26.67','40','80']

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



   if len(subjs) > 1:
      ss_str='subjs'
   else:
      ss_str=subjs[0]

   if dayblock.find('day2') > 0 or dayblock.find('day3') > 0:
      f_bf=8
   else:
      f_bf=35

   if dayblock.find('itr') >=0:

      probs_iti_per_s=np.empty((len(compr),len(subjs),n_seqs))
      probs_iti_per_s[:]=np.nan
      probs_iti_per_s_per_trial=np.empty((len(compr),len(subjs),bf+1,n_seqs))
      probs_iti_per_s_per_trial[:]=np.nan
      win_seq_iti=[[[] for jj in subjs] for ii in compr]
      src_iti = [[[] for jj in subjs] for ii in compr]
      #src_iti = [ np.empty((len(subjs),), dtype=np.object) for ii in compr]

      #pool = mp.Pool(processes=njobs)
      for ii in itertools.product(enumerate(compr),enumerate(subjs)):

         ind_cc,cc=ii[0]
         ind_ss,ss=ii[1]
         
         filename='/scratch/claudinolm/%s/%s.0.%d.%sx.%s.pkl'%(indir,ss,f_bf,cc,dayblock)

         log_result(score_itr(ind_cc,cc,ind_ss,ss,n_seqs,b0,bf,dayblock,thr,filename))
         #pool.apply_async(score_itr,(ind_cc,cc,ind_ss,ss,n_seqs,b0,bf,dayblock,thr,filename),callback=log_result)

      #pool.close()
      #pool.join()

      #np.save('/scratch/claudinolm/%s/%s.%.4f.probs_%s_per_s'%(outdir,ss_str,thr,dayblock),probs_iti_per_s)
      #np.save('/scratch/claudinolm/%s/%s.%.4f.probs_%s_per_s_per_trial'%(outdir,ss_str,thr,dayblock),
      #probs_iti_per_s_per_trial)
      #np.save('/scratch/claudinolm/%s/%s.%.4f.%s.win'%(outdir,ss_str,thr,dayblock),win_seq_iti)

      # Find matrix size at each compression rate to later create empyt array
      '''
      sizes=[(1,1) for ii in compr]
      for ind_ii,ii in enumerate(src_iti):
         for jj in ii:
            #print len(jj)
            #print jj.shape
            if len(jj) > 0:
               sizes[ind_ii] = jj.shape
               break

      data = [[[] for jj in subjs] for ii in compr]
      for ind_ii,ii in enumerate(src_iti):
         print ind_ii
         for ind_jj,jj in enumerate(ii):
            if len(jj) > 0:
               try:
                  data[ind_ii][ind_jj]=jj.astype(np.object)
               except:
                  pdb.set_trace()

            else:
               try:
                  out=np.empty(sizes[ind_ii],dtype=np.object)
                  out[:]=np.nan
                  data[ind_ii][ind_jj]=out
               except:
                  pdb.set_trace()
      savemat('/scratch/claudinolm/%s/%s.%.4f.%s.%d.%d.source'%(outdir,ss_str,thr,dayblock,b0,bf),
      {'%s_src'%args.dayblock:data,'subjs':subjs,'seq':[4,1,3,2,4]})
      '''

      savemat('/scratch/claudinolm/%s/%s.%.4f.%s.%d.%d.source'%(outdir,ss_str,thr,dayblock,b0,bf),
      {'%s_src'%args.dayblock:src_iti,'subjs':subjs,'seq':[4,1,3,2,4]})

   else:
      probs_pre_per_s=np.empty((len(compr),len(subjs),n_seqs))
      probs_pre_per_s[:]=np.nan      
      probs_pre_per_s_ma=[[[] for jj in subjs] for ii in compr]
      win_seq_pre=[[[] for jj in subjs] for ii in compr]
      src_pre = [[[] for jj in subjs] for ii in compr]
      #source_pre_all=[[]]*len(compr)

      #pool = mp.Pool(processes=njobs)
      for ii in itertools.product(enumerate(compr),enumerate(subjs)):

         ind_cc,cc=ii[0]
         ind_ss,ss=ii[1]

         # NAMING OF REST FILES HAS ALWAYS B0=0 AND BF=35, FIX THIS
         filename='/scratch/claudinolm/%s/%s.0.35.%sx.%s.pkl'%(indir,ss,cc,dayblock)

         log_result(score_rest(ind_cc,cc,ind_ss,ss,n_seqs,dayblock,thr,filename))
         #pool.apply_async(score_rest,(ind_cc,cc,ind_ss,ss,n_seqs,dayblock,thr,filename),callback=log_result)

      #pool.close()
      #pool.join()

      #np.save('/scratch/claudinolm/%s/%s.%.4f.probs_%s_per_s'%(outdir,ss_str,thr,dayblock),probs_pre_per_s)
      #joblib.dump(probs_pre_per_s_ma,'/scratch/claudinolm/%s/%s.%.4f.probs_%s_per_s_ma.pkl'%(outdir,ss_str,thr,dayblock))
      #np.save('/scratch/claudinolm/%s/%s.%.4f.%s.win'%(outdir,ss_str,thr,dayblock),win_seq_pre)
      savemat('/scratch/claudinolm/%s/%s.%.4f.%s.source'%(outdir,ss_str,thr,dayblock),
      {'%s_src'%args.dayblock:src_pre,'subjs':subjs,'seq':[4,1,3,2,4]})

   #np.save('/scratch/claudinolm/%s/compr'%outdir,compr)


if __name__=='__main__':

   # Find all subject codes from meg.txt file
   all_subjs=re.findall('([A-Z]+)\.day\d\/.+\n',open('meg.txt').read())
   all_subjs=list(set(all_subjs))

   # Parse input arguments
   parser = argparse.ArgumentParser(description='Test key press decoding')
   parser.add_argument('--meg_codes',dest='subjs',default=all_subjs,nargs='+')
   parser.add_argument('--njobs',dest='njobs',type=int)
   parser.add_argument('--b0',dest='b0',type=int)
   parser.add_argument('--bf',dest='bf',type=int)
   parser.add_argument('--dayblock',dest='dayblock',type=str)
   parser.add_argument('--dir',dest='dir',type=str,default='seq_decoder_broad_per_day')
   parser.add_argument('--thr',dest='thr',type=float)
   parser.add_argument('--pickseqs',dest='pickseqs',default='',type=str)
   parser.add_argument('--seq',dest='target',type=str)
   args=parser.parse_args()

   if args.pickseqs <> '':
      n_seqs= len([tuple(int(jj) for jj in ii) for ii in re.findall('(\d+)',open('seqs.txt','rt').read())])
   else:
      n_seqs=1024

   outdir=args.dir.replace('decoder','results')

   #np.save('/scratch/claudinolm/%s/subjs'%outdir,args.subjs)

   control(args.subjs,args.njobs, args.dayblock, args.b0, args.bf, args.thr, n_seqs, args.dir, outdir)

