import numpy as np, matplotlib.pyplot as plt, cPickle as pickle
import mne, sys, os, pdb, re, argparse, itertools,csv,glob
from mne.beamformer import make_lcmv, apply_lcmv_epochs, apply_lcmv_raw
from mne.source_space import get_volume_labels_from_src

if __name__ == '__main__':

   # Parse input arguments
   all_subjs=re.findall('([A-Z]+)\.day\d\/.+\n',open('meg.txt').read())
   all_subjs=list(set(all_subjs))

   parser = argparse.ArgumentParser(description='Inverse model at each t.')
   parser.add_argument('--meg_codes',dest='subjs',default=all_subjs,nargs='+')
   parser.add_argument('--njobs', dest='njobs',type=int,default=1)
   parser.add_argument('--lo',dest='lo',type=float,default=1)
   parser.add_argument('--hi',dest='hi',type=float,default=100)

   args=parser.parse_args()

   for ss in args.subjs:

      roi_labels = mne.read_labels_from_annot(ss,'BN_Atlas')

      # Compute noise cov matrix (day1 rest first 20s)
      raw_day1rest=mne.io.read_raw_fif('coreg/%s.day1rest.coreg.ica.%.2f-%.2fHz_raw.fif'\
      %(ss,args.lo,args.hi), preload=True)

      noise_cov = mne.compute_raw_covariance(raw_day1rest, tmin=0, tmax=19.999,
      n_jobs=args.njobs, method='empirical')

      ############################################################
      try:

         if not os.path.exists('kp_source_broad/%s.restday1.X.npy'%ss):
            
            # Read fwd solution
            bem_folder='%s/%s/bem'%(os.environ['SUBJECTS_DIR'],ss)
            fwd = mne.read_forward_solution('%s/%s_fwd.day1rest.fif'%(bem_folder,ss))
            fwd = mne.convert_forward_solution(fwd, surf_ori=True)

            #lut_file='/data/claudinolm/nrsa/seq_learning/freesurfer/subjects/BN_Atlas_246_LUT.txt'
            #vol_labels = get_volume_labels_from_src(fwd['src'],ss, os.environ['SUBJECTS_DIR'],
            #'BN_Atlas_subcortex.mgz',lut_file)

            raw_day1rest.crop(tmin=20, tmax=None)

            raw_day1rest=raw_day1rest.pick_types(meg=True,stim=False,eeg=False,\
            ref_meg=True,eog=False,emg=False,exclude='bads')

            restday1_cov  = mne.compute_raw_covariance(raw_day1rest, n_jobs=args.njobs, method='empirical')
            #raw_day1rest.info['comps']=[]

            lcmv_restday1 =make_lcmv(raw_day1rest.info, forward=fwd, noise_cov=noise_cov, data_cov=restday1_cov)
            stc_restday1 = apply_lcmv_raw(raw_day1rest, lcmv_restday1)
            stc_restday1.resample(sfreq=200, n_jobs=args.njobs, npad=0)

            restday1_ts = mne.extract_label_time_course([stc_restday1], roi_labels, fwd['src'],
            allow_empty=True, mode='mean_flip')

            np.save('kp_source_broad/%s.restday1.X'%ss, np.vstack(restday1_ts).T)
      
      except:
         print 'Could not do restday1 on %s'%ss
         pass

      ############################################################
      try:

         if not os.path.exists('kp_source_broad/%s.restday1pos.X.npy'%ss):

            # Read fwd solution
            bem_folder='%s/%s/bem'%(os.environ['SUBJECTS_DIR'],ss)
            fwd = mne.read_forward_solution('%s/%s_fwd.day1restpos.fif'%(bem_folder,ss))
            fwd = mne.convert_forward_solution(fwd, surf_ori=True)

            raw_day1restpos=mne.io.read_raw_fif('coreg/%s.day1restpos.coreg.ica.%.2f-%.2fHz_raw.fif'\
            %(ss,args.lo,args.hi), preload=True)

            raw_day1restpos=raw_day1restpos.pick_types(meg=True,stim=False,eeg=False,\
            ref_meg=True,eog=False,emg=False,exclude='bads')

            restday1pos_cov  = mne.compute_raw_covariance(raw_day1restpos, n_jobs=args.njobs, method='empirical')
            #raw_day1restpos.info['comps']=[]

            lcmv_restday1pos =make_lcmv(raw_day1restpos.info, forward=fwd, noise_cov=noise_cov, data_cov=restday1pos_cov)
            stc_restday1pos = apply_lcmv_raw(raw_day1restpos, lcmv_restday1pos)
            stc_restday1pos.resample(sfreq=200, n_jobs=args.njobs, npad=0)
            restday1pos_ts = mne.extract_label_time_course([stc_restday1pos], roi_labels, fwd['src'],
            allow_empty=True, mode='mean_flip')

            np.save('kp_source_broad/%s.restday1pos.X'%ss, np.vstack(restday1pos_ts).T)
      except:
         print 'Could not do restday1pos on %s'%ss
         pass

      ############################################################
      try:

         if not os.path.exists('kp_source_broad/%s.restday2.X.npy'%ss):

            # Read fwd solution
            bem_folder='%s/%s/bem'%(os.environ['SUBJECTS_DIR'],ss)
            fwd = mne.read_forward_solution('%s/%s_fwd.day2rest.fif'%(bem_folder,ss))
            fwd = mne.convert_forward_solution(fwd, surf_ori=True)

            raw_day2rest=mne.io.read_raw_fif('coreg/%s.day2rest.coreg.ica.%.2f-%.2fHz_raw.fif'\
            %(ss,args.lo,args.hi), preload=True)


            raw_day2rest=raw_day2rest.pick_types(meg=True,stim=False,eeg=False,\
            ref_meg=True,eog=False,emg=False,exclude='bads')

            restday2_cov  = mne.compute_raw_covariance(raw_day2rest, n_jobs=args.njobs, method='empirical')
            #raw_day2rest.info['comps']=[]

            lcmv_restday2 =make_lcmv(raw_day2rest.info, forward=fwd, noise_cov=noise_cov, data_cov=restday2_cov)
            stc_restday2 = apply_lcmv_raw(raw_day2rest, lcmv_restday2)
            stc_restday2.resample(sfreq=200, n_jobs=args.njobs, npad=0)
            restday2_ts = mne.extract_label_time_course([stc_restday2], roi_labels, fwd['src'],
            allow_empty=True, mode='mean_flip')

            np.save('kp_source_broad/%s.restday2.X'%ss, np.vstack(restday2_ts).T)
      
      except:
         print 'Could not do restday2 on %s'%ss
         pass
      ############################################################
      try:

         if not os.path.exists('kp_source_broad/%s.restday2pos.X.npy'%ss):

            # Read fwd solution
            bem_folder='%s/%s/bem'%(os.environ['SUBJECTS_DIR'],ss)
            fwd = mne.read_forward_solution('%s/%s_fwd.day2restpos.fif'%(bem_folder,ss))
            fwd = mne.convert_forward_solution(fwd, surf_ori=True)

            raw_day2restpos=mne.io.read_raw_fif('coreg/%s.day2restpos.coreg.ica.%.2f-%.2fHz_raw.fif'\
            %(ss,args.lo,args.hi), preload=True)

            raw_day2restpos=raw_day2restpos.pick_types(meg=True,stim=False,eeg=False,\
            ref_meg=True,eog=False,emg=False,exclude='bads')

            restposday2_cov  = mne.compute_raw_covariance(raw_day2restpos, n_jobs=args.njobs, method='empirical')
            #raw_day2restpos.info['comps']=[]

            lcmv_restposday2 =make_lcmv(raw_day2restpos.info, forward=fwd, noise_cov=noise_cov, data_cov=restposday2_cov)
            stc_restposday2 = apply_lcmv_raw(raw_day2restpos, lcmv_restposday2)
            stc_restposday2.resample(sfreq=200, n_jobs=args.njobs, npad=0)
            restposday2_ts = mne.extract_label_time_course([stc_restposday2], roi_labels, fwd['src'],
            allow_empty=True, mode='mean_flip')

            np.save('kp_source_broad/%s.restday2pos.X'%ss, np.vstack(restposday2_ts).T)
      
      except:
         print 'Could not do restposday2 on %s'%ss
         pass
      ############################################################
      try:

         if not os.path.exists('kp_source_broad/%s.restday3.X.npy'%ss):

            # Read fwd solution
            bem_folder='%s/%s/bem'%(os.environ['SUBJECTS_DIR'],ss)
            fwd = mne.read_forward_solution('%s/%s_fwd.day3rest.fif'%(bem_folder,ss))
            fwd = mne.convert_forward_solution(fwd, surf_ori=True)

            raw_day3rest=mne.io.read_raw_fif('coreg/%s.day3rest.coreg.ica.%.2f-%.2fHz_raw.fif'\
            %(ss,args.lo,args.hi), preload=True)

            raw_day3rest=raw_day3rest.pick_types(meg=True,stim=False,eeg=False,\
            ref_meg=True,eog=False,emg=False,exclude='bads')

            restday3_cov  = mne.compute_raw_covariance(raw_day3rest, n_jobs=args.njobs, method='empirical')
            #raw_day3restpos.info['comps']=[]

            lcmv_restday3 =make_lcmv(raw_day3rest.info, forward=fwd, noise_cov=noise_cov, data_cov=restday3_cov)
            stc_restday3 = apply_lcmv_raw(raw_day3rest, lcmv_restday3)
            stc_restday3.resample(sfreq=200, n_jobs=args.njobs, npad=0)
            restday3_ts = mne.extract_label_time_course([stc_restday3], roi_labels, fwd['src'],
            allow_empty=True, mode='mean_flip')

            np.save('kp_source_broad/%s.restday3.X'%ss, np.vstack(restday3_ts).T)
      
      except:
         print 'Could not do restday3 on %s'%ss
         pass
      ############################################################
      try:

         if not os.path.exists('kp_source_broad/%s.restday3pos.X.npy'%ss):

            # Read fwd solution
            bem_folder='%s/%s/bem'%(os.environ['SUBJECTS_DIR'],ss)
            fwd = mne.read_forward_solution('%s/%s_fwd.day3restpos.fif'%(bem_folder,ss))
            fwd = mne.convert_forward_solution(fwd, surf_ori=True)

            raw_day3restpos=mne.io.read_raw_fif('coreg/%s.day3restpos.coreg.ica.%.2f-%.2fHz_raw.fif'\
            %(ss,args.lo,args.hi), preload=True)


            raw_day3restpos=raw_day3restpos.pick_types(meg=True,stim=False,eeg=False,\
            ref_meg=True,eog=False,emg=False,exclude='bads')

            restposday3_cov  = mne.compute_raw_covariance(raw_day3restpos, n_jobs=args.njobs, method='empirical')
            #raw_day3restpos.info['comps']=[]

            lcmv_restposday3 =make_lcmv(raw_day3restpos.info, forward=fwd, noise_cov=noise_cov, data_cov=restposday3_cov)
            stc_restposday3 = apply_lcmv_raw(raw_day3restpos, lcmv_restposday3)
            stc_restposday3.resample(sfreq=200, n_jobs=args.njobs, npad=0)
            restposday3_ts = mne.extract_label_time_course([stc_restposday3], roi_labels, fwd['src'],
            allow_empty=True, mode='mean_flip')

            np.save('kp_source_broad/%s.restday3pos.X'%ss, np.vstack(restposday3_ts).T)
      
      except:
         print 'Could not do restposday3 on %s'%ss
         pass

