import numpy as np, matplotlib.pyplot as plt, cPickle as pickle
import mne, sys, os, pdb, re, argparse, itertools,csv,glob
from mne.beamformer import make_lcmv, apply_lcmv_epochs, apply_lcmv_raw


if __name__ == '__main__':
   
   # Parse input arguments
 
   # Create a default list with any subject code in the meg.txt file
   all_subjs=re.findall('([A-Z]+)\.day\d\/.+\n',open('meg.txt').read())
   all_subjs=list(set(all_subjs))

   parser = argparse.ArgumentParser(description='Inverse model at each t.')
   parser.add_argument('--meg_codes',dest='subjs',default=all_subjs,nargs='+')
   parser.add_argument('--blocks',dest='blocks',nargs='+')
   parser.add_argument('--njobs', dest='njobs',type=int,default=1)
   parser.add_argument('--lo',dest='lo',type=float,default=1)
   parser.add_argument('--hi',dest='hi',type=float,default=100)

   args=parser.parse_args()

   for ss,bb in itertools.product(args.subjs, args.blocks):

      if 'day1' in bb:
         bb_day='day1'
         max_evs=36
      elif 'day2' in bb:
         bb_day='day2'
         max_evs=9
      elif 'day3' in bb:
         bb_day='day3'
         max_evs=9

      roi_labels = mne.read_labels_from_annot(ss,'BN_Atlas')

      bem_folder='%s/%s/bem'%(os.environ['SUBJECTS_DIR'],ss)
      fwd = mne.read_forward_solution('%s/%s_fwd.%s.fif'%(bem_folder,ss,bb))
      fwd = mne.convert_forward_solution(fwd, surf_ori=True)

      # Open raw
      raw=mne.io.read_raw_fif('coreg/%s.%s.coreg.ica.%.2f-%.2fHz_raw.fif'\
      %(ss,bb,args.lo,args.hi), preload=True)

      picks=mne.pick_types(raw.info, meg=True,stim=False,eeg=False,\
      ref_meg=True,eog=False,emg=False,exclude='bads')

      # Compute noise cov matrix (always from day1rest)
      raw_day1rest=mne.io.read_raw_fif('coreg/%s.day1rest.coreg.ica.%.2f-%.2fHz_raw.fif'\
      %(ss,args.lo,args.hi), preload=True)

      noise_cov = mne.compute_raw_covariance(raw_day1rest, tmin=0, tmax=19.999,
      n_jobs=args.njobs, method='empirical')

      # Epoch the data into trials
      events=mne.find_events(raw,initial_event=True)

      if events.shape[0] < max_evs:
          ind_events = [int(ii) for ii in open('etc/%s.%s.valid_trials.txt'%(ss,bb_day)).read().split()] 
      else:
         ind_events=range(max_evs)

      # The code goes over the CSV with behavior events, and if an event is in the valid trials file
      # the corresponding MEG index is obtained
      # Eg: 
      # 1, 2, 3, 8 --> valid_trials (to be looked up in the CSV)
      # 0, 1, 2, 3 --> corresp events detected by MEG (always same number as valid trials)
      #
      # In this case, valid behav 8 = MEG 3 and so on

      csv_list  = list(csv.reader(open('behav/%s_%s_events.csv'%(bb_day,ss))))[1:]

      offset={ii[0]: (int(ii[3]),events[ind_events.index(int(ii[0])-1),0])\
      for ii in csv_list if ii[2] == 'NaN' and int(ii[0])-1 in ind_events}

      kp_times=[((int(ii[3])-offset[ii[0]][0])*0.6 + offset[ii[0]][1],\
      ind_events.index(int(ii[0])-1),ii[1])\
         for ii in csv_list if ii[2] == '1']

      seq_end_times=[((int(ii[3])-offset[ii[0]][0])*0.6 + offset[ii[0]][1],\
      ind_events.index(int(ii[0])-1))\
      for ind,ii in enumerate(csv_list) if ii[5]=='1']

      seq_beg_times=[((int(ii[3])-offset[ii[0]][0])*0.6 + offset[ii[0]][1])\
      for ind,ii in enumerate(csv_list) if ind+5 <= len(csv_list) and ii[1]=='4'\
      and csv_list[ind+4][5] == '1']

      [seq_times,seq_evs]=zip(*seq_end_times)

      seq_events = np.zeros((len(seq_times),3),dtype=int)
      seq_events[:,0]=seq_times
      seq_events[:,2]=seq_evs
      seq_events=seq_events[seq_events[:,0].argsort()]
      seq_evs=seq_events[:,2]

      # To extract play sequences
      seq_delta_times=np.array(seq_times,dtype=int)-np.array(seq_beg_times,dtype=int)

      seq_end_epochs = mne.Epochs(raw,seq_events,tmin=-max(seq_delta_times)/raw.info['sfreq'],
      tmax=0,reject=None,flat=None,baseline=None, picks=picks,preload=True)

      # Epoch keypresses
      [times,evs,keys]=zip(*kp_times)

      # Looks for and remove events out of order. These are artefacts
      # of simultaneous keypresses

      kp_events=np.zeros((len(times),4),dtype=int)
      kp_events[:,0]=times
      kp_events[:,2]=keys
      kp_events[:,3]=evs
      kp_events=kp_events[kp_events[:,0].argsort()]

      # Remove events with repeated timestamps
      remove=np.unique([(kk,kk+1) for kk in np.where(np.diff(kp_events[:,0])==0)])
      kp_events=np.delete(kp_events,remove,0)

      evs=kp_events[:,3]
      kp_events=kp_events[:,0:3]

      kp_epochs=mne.Epochs(raw,kp_events,tmin=-0.5,tmax=0.5,
      reject=None,flat=None,baseline=None, picks=picks,preload=True)

      # Will calculate one lcmv per trial.
      epochs=mne.Epochs(raw,events,tmin=0,tmax=20,reject=None,flat=None,
      baseline=(None,None), picks=picks,preload=True)

      ####################################################################################################
      # Neural patterns of entire sequences
      ####################################################################################################
      for ii in np.unique(seq_evs): 
            
         print 'LCMV seq at epoch %d'%ii
         epoch_motor=epochs[ii].copy().crop(tmin=0,tmax=9.998)
         motor_cov = mne.compute_covariance(epoch_motor, n_jobs=args.njobs, method='empirical')
         lcmv_motor=make_lcmv(epoch_motor.info, forward=fwd, noise_cov=noise_cov,data_cov=motor_cov)

         # Apply source to the seqs associated with current experiment epoch      

         stc_motor=apply_lcmv_epochs(seq_end_epochs[seq_evs==ii],lcmv_motor)
         # -1 below is because the last event is a 0 between motor and rest

         # Play
         data_play=[]
         for jj,kk in zip(stc_motor,seq_delta_times[seq_evs==ii]): 
            print 'Extracting sources block %d, keypress %d'%(ii,kk) 
            start=jj.times[jj.times.shape[0]-kk+1]

            jj_tmp=jj.copy()
            jj_tmp.crop(tmin=start,tmax=0)
            jj_tmp.resample(sfreq=500, n_jobs=args.njobs, npad=0)

            data_ts = mne.extract_label_time_course([jj_tmp], roi_labels, fwd['src'], 
            allow_empty=True, mode='mean_flip')

            data_play.append(np.vstack(data_ts))


         ii_evs=ind_events[ii]
         pickle.dump(data_play,open('kp_source_broad/%s.%s.%.2d.motor.epochs.play.pkl'%(ss,bb,ii_evs),'wb'))

      ####################################################################################################
      # Key presses
      ####################################################################################################
      for ii in np.unique(evs): 
            
         print 'LCMV at epoch %d'%ii

         epoch_motor=epochs[ii].copy().crop(tmin=0,tmax=9.998)
         epoch_rest =epochs[ii].copy().crop(tmin=10.002,tmax=20)
         motor_cov = mne.compute_covariance(epoch_motor, n_jobs=args.njobs, method='empirical')
         rest_cov  = mne.compute_covariance(epoch_rest, n_jobs=args.njobs, method='empirical')

         lcmv_motor=make_lcmv(epoch_motor.info, forward=fwd, noise_cov=noise_cov,data_cov=motor_cov)
         lcmv_rest =make_lcmv(epoch_rest.info, forward=fwd, noise_cov=noise_cov,data_cov=rest_cov)

         # Apply source to the key presses associated with current experiment epoch      

         stc_motor=apply_lcmv_epochs(kp_epochs[evs==ii],lcmv_motor)
         # -1 below is because the last event is a 0 between motor and rest
         data=[]
         for jj in stc_motor[:-1]: 
            jj.resample(sfreq=500, n_jobs=args.njobs, npad=0)

            data_ts = mne.extract_label_time_course([jj], roi_labels, fwd['src'], 
            allow_empty=True, mode='mean_flip')
            
            data.append(np.vstack(data_ts))


         data=np.asarray(data)

         stc_rest=apply_lcmv_epochs(epoch_rest, lcmv_rest)
         stc_rest[0].resample(sfreq=500, n_jobs=args.njobs, npad=0)
         rest_ts = mne.extract_label_time_course([stc_rest[0]], roi_labels, fwd['src'], 
         allow_empty=True, mode='mean_flip')
 
         stc_motor=apply_lcmv_epochs(epoch_motor, lcmv_motor)
         stc_motor[0].resample(sfreq=500, n_jobs=args.njobs, npad=0)
         motor_ts = mne.extract_label_time_course([stc_motor[0]], roi_labels, fwd['src'], 
         allow_empty=True, mode='mean_flip')

         ii_evs=ind_events[ii]
         np.save('kp_source_broad/%s.%s.%.2d.motor.epochs.X'%(ss,bb,ii_evs),data)
         np.save('kp_source_broad/%s.%s.%.2d.motor.epochs.y'%(ss,bb,ii_evs), kp_events[evs==ii,2][:-1])
         np.save('kp_source_broad/%s.%s.%.2d.rest.X'%(ss,bb,ii_evs), np.vstack(rest_ts).T)
         np.save('kp_source_broad/%s.%s.%.2d.motor.X'%(ss,bb,ii_evs), np.vstack(motor_ts).T)

