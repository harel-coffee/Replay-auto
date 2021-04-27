import mne, os, re, itertools, sys, argparse, pdb
import numpy as np, matplotlib.pyplot as plt
from mne.preprocessing import ICA, read_ica

# Find all subject codes from meg.txt file
all_subjs=re.findall('([A-Z]+)\.day1\/.+',open('meg.txt').read())

# Parse input arguments
parser = argparse.ArgumentParser(description='ICA helper.')
parser.add_argument('--meg_codes',dest='subjs',default=all_subjs,nargs='+')
parser.add_argument('--blocks',dest='blocks',nargs='+')
parser.add_argument('--inds',dest='inds',default=[],nargs='+')
parser.add_argument('--viz_inds',dest='inds',default=[],nargs='+',type=int)
parser.add_argument('--bad_channels',dest='bad_channels',nargs="+",default=[])
parser.add_argument('--lo',dest='lo',type=float,default=0)
parser.add_argument('--hi',dest='hi',type=float,default=0)
parser.add_argument('--t0',dest='t0',type=float, default=-1)
parser.add_argument('--tf',dest='tf',type=float, default=-1)
parser.add_argument('--ncomps',dest='ncomps',type=int,default=10)
parser.add_argument('--njobs', dest='njobs',type=int,default=1)
parser.add_argument('--viz',action='store_true')
parser.add_argument('--redo',action='store_true')
parser.add_argument('--clean',action='store_true')

args=parser.parse_args()

if args.inds[0] == 'all':
   args.inds=range(args.ncomps)
else:
   args.inds = [int(ii) for ii in args.inds]

# Will go over all selected blocks of all selected subjects serially
for ss,bb in itertools.product(args.subjs, args.blocks):

    # Fetch file name from block and subject information
    try:
       filename = re.findall('%s.%s/(.+)'%(ss,bb),open('meg.txt').read())[0].strip()
    except:
       print 'Did not find subj %ss, cond %s'%(ss, bb)
       continue

    # Form output ICA object and filtered-ICAed file
    out_ica='ica/%s.%s_ica.fif'%(ss,bb)
    out_filt='filt/%s.%s.%.2f-%.2fHz_raw.fif'%(ss,bb,args.lo,args.hi)
    out_ica_filt='filt/%s.%s.ica.%.2f-%.2fHz_raw.fif'%(ss,bb,args.lo,args.hi)
    
    # Will run if ICA has not been previously computed
    if not os.path.exists(out_ica) or args.redo:

       # Try to open a CTF file, next try to open a FIF
       try:
          raw=mne.io.read_raw_ctf('meg/%s'%filename, preload=True)
       except:
          raw=mne.io.read_raw_fif('meg/%s'%filename, preload=True)

       # Mark bad channels and select MEG and compensation
       raw.info['bads']=args.bad_channels
       picks=mne.pick_types(raw.info, meg=True,stim=False,eeg=False,
       ref_meg=False,eog=False,emg=False,exclude='bads')

       # Crop first and filter later to avoid artifacts from spreading into
       # the time window of interest
       if args.t0 > 0 and args.tf > 0:
          raw.crop(tmin=args.t0, tmax=args.tf)

       # Band pass and notch filter (if applicable)
       if args.lo > 0 and args.hi > 0:
          raw.filter(args.lo,args.hi, picks, n_jobs=args.njobs,
          fir_design='firwin')
       if args.hi >= 60:
          raw.notch_filter(freqs=60,n_jobs=args.njobs)

       # Calculate (fast) ICA
       try:
          events=mne.find_events(raw)
          epochs=mne.Epochs(raw,events,reject=None,
          flat=None,baseline=None,tmin=0,tmax=20)
       except:
          epochs=raw

       ica = ICA(n_components=args.ncomps, method='fastica',
       random_state=11901,max_iter=500)
       ica.fit(epochs, picks=picks)
       ica.save(out_ica)
       raw.save(out_filt,overwrite=True)
	
    # Will either visualize or clean the data
    else:
       # Read ICA solution
       ica = read_ica(out_ica)


       try:
      
          # Read bandpassed filtered data that ICA was calculated from
          raw=mne.io.read_raw_fif(out_filt, preload=True)

       except:

          # Try to open a CTF file, next try to open a FIF
          try:
             raw=mne.io.read_raw_ctf('meg/%s'%filename, preload=True)
          except:
             raw=mne.io.read_raw_fif('meg/%s'%filename, preload=True)

          # Mark bad channels and select MEG and compensation
          raw.info['bads']=args.bad_channels
          picks=mne.pick_types(raw.info, meg=True,stim=False,eeg=False,
          ref_meg=False,eog=False,emg=False,exclude='bads')

          # Crop first and filter later to avoid artifacts from spreading into
          # the time window of interest
          if args.t0 > 0 and args.tf > 0:
             raw.crop(tmin=args.t0, tmax=args.tf)

          # Band pass and notch filter (if applicable)
          if args.lo > 0 and args.hi > 0:
             raw.filter(args.lo,args.hi, picks, n_jobs=args.njobs,
             fir_design='firwin')
          if args.hi >= 60:
             raw.notch_filter(freqs=60,n_jobs=args.njobs)

          raw.save(out_filt,overwrite=True)

       
       # Visualize
       if args.viz:
          
          # Show timeseries plot
          ica.plot_sources(raw, picks=args.inds, block=True)
          #plt.show(False)
          

          # Show time series, psds and topoplots of each IC
          #ica.plot_properties(raw,picks=args.inds, 
          #psd_args={'fmin':args.lo,'fmax': args.hi,'n_jobs': args.njobs}, 
          #topomap_args={'outlines':'skirt'} )

       # Clean
       if args.clean:
          clean=ica.apply(raw,exclude=args.inds)
          clean.save(out_ica_filt,overwrite=True)
       elif not args.viz: # No cleaning, just copying under diff name to fit with pipeline
          raw.save(out_ica_filt)

