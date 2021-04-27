import numpy as np, matplotlib.pyplot as plt
from scipy.stats import sem, pearsonr, spearmanr,ttest_rel,zscore
from scipy.io import savemat
from numpy.polynomial.polynomial import polyfit, polyval
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor
from sklearn.externals import joblib
import pdb, argparse, re, itertools, h5py, hdf5storage
from numpy import ma


if __name__=='__main__':

   parser = argparse.ArgumentParser(description='Generate time resolved rates for rest blocks')
   parser.add_argument('--pickseqs',dest='pickseqs',default='',type=str)
   parser.add_argument('--seq',dest='target',type=str)
   parser.add_argument('--dir',dest='dir',type=str,default='seq_decoder_broad_per_day')
   parser.add_argument('--thr',dest='thr',type=float)
   parser.add_argument('--dayblock',dest='dayblock',type=str)
   args=parser.parse_args()

   # Parse input arguments
   subjs = np.load('/scratch/claudinolm/%s/subjs.npy'%args.dir).tolist()
   compr = np.load('/scratch/claudinolm/%s/compr.npy'%args.dir)

   outdir=args.dir.replace('decoder','results')

   if args.pickseqs <> '':
      perms  = [tuple(int(jj) for jj in ii) for ii in re.findall('(\d+)',open('seqs.txt','rt').read())]
   else:
      perms=list(itertools.product([1,2,3,4],repeat=5))
   ind_seq = [ind for ind,ii in enumerate(perms) if str(ii)[1:-1].replace(', ','') == args.target][0]
   

   counts_all=np.empty((len(subjs),), dtype=np.object)
   if args.dayblock.find('itr') >=0:
      print 'ITR'
      '''
      #counts=np.load('/scratch/claudinolm/%s/subjs.%.4f.probs_%s_per_s_per_trial.npy'%(args.dir,args.thr,args.dayblock))[:,:,:,ind_seq]
      counts_all=np.load('/scratch/claudinolm/%s/subjs.%.4f.probs_%s_per_s_per_trial.npy'%(args.dir,args.thr,args.dayblock))
      '''

      #counts_subj=joblib.load('/scratch/claudinolm/%s/%s.%.4f.probs_%s_per_s_ma.pkl'\
      #%(args.dir,ss,args.thr,args.dayblock))


      counts_all=joblib.load('/scratch/claudinolm/%s/subjs.%.4f.probs_%s_per_s_ma.pkl'\
      %(args.dir,args.thr,args.dayblock))


      for ind_ss,ss in enumerate(subjs):
         
         print 'Adding time-resolved to mat for subj %s'%ss
         #counts_all[ind_ss] = counts[:,ind_ss,:]
         
         counts_subj=np.array([ii[ind_ss] for ii in counts_all])
   
         data={u'%s_tr'%args.dayblock:counts_subj, u'subjs':subjs, u'seq':perms}
         
         hdf5storage.write(data=data, 
         filename='/scratch/claudinolm/%s/%s_%s_raster.mat'%(outdir,ss,args.dayblock),matlab_compatible=True)

      '''
      for ind_ss,ss in enumerate(subjs):
         print 'Adding time-resolved to mat for subj %s'%ss
         counts_all[ind_ss] = counts[:,ind_ss,:]
      '''

   else:
      print 'Not ITR'

      for ind_ss,ss in enumerate(subjs):

         counts_subj=joblib.load('/scratch/claudinolm/%s/%s.%.4f.probs_%s_per_s_ma.pkl'\
         %(args.dir,ss,args.thr,args.dayblock))

         data={u'%s_tr'%args.dayblock:counts_subj, u'subjs':subjs, u'seq':perms}

         print ss
         hdf5storage.write(data=data,
         filename='/scratch/claudinolm/%s/%s_%s_raster.mat'%(outdir,ss,args.dayblock),matlab_compatible=True)

         '''
         print 'Adding time-resolved to mat for subj %s'%ss
         #counts_all[ind_ss]=counts_subj[:][0][ind_seq]
         if len(counts_subj[0][0]) > 0:
            #counts_all[ind_ss]=[ii[0][ind_seq] for ii in counts_subj]
            counts_all[ind_ss]=np.squeeze(counts_subj,1)
         else:
            counts_all[ind_ss]=np.nan
         '''

   '''
   data={u'%s_tr'%args.dayblock:counts_all, u'subjs':subjs, u'seq':perms}

   pdb.set_trace()

   hdf5storage.write(data=data,
   filename='/scratch/claudinolm/%s/%s_time_resolved.mat'%(outdir,args.dayblock),matlab_compatible=True)
   '''

