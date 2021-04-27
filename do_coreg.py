import re, mne, itertools, argparse, os, pdb
import numpy as np, subprocess as sp
from mayavi import mlab

# Find all subject codes from meg.txt file
all_subjs=re.findall('([A-Z]+)\.day\d\/.+\n',open('meg.txt').read())
all_subjs=list(set(all_subjs))

# Parse input arguments
parser = argparse.ArgumentParser(description='Coregistration.')
parser.add_argument('--meg_codes',dest='subjs',default=all_subjs,nargs='+')
parser.add_argument('--blocks',dest='blocks',nargs='+')
parser.add_argument('--lo',dest='lo',type=float)
parser.add_argument('--hi',dest='hi',type=float)
parser.add_argument('--viz',action='store_true')

args=parser.parse_args()

# Will go over all selected blocks of all selected subjects serially
for ss,bb in itertools.product(args.subjs, args.blocks):

   if 'day1' in bb or 'Day1' in bb:
      day=1
   elif 'day2' in bb or 'Day2' in bb: 
      day=2
   elif 'day3' in bb or 'Day3' in bb: 
      day=3

   # Find fiducials in MRI coordinates by reading from brainsight (ELP)
   try:
      # Original data format
      if day <> 3:
         mrifid = open('neuronav/%s/CoilPosD%d.txt'%(ss,day)).read()
         ind_fid=0
      else:
         mrifid = open('neuronav/%s/%s_CoilPosAll.txt'%(ss,ss)).read()
         ind_fid=2

      nasion=[float(ii) for ii in re.findall('NEC\sSession\s[\d+|"(null)"\s]+[\d+|"(null)"]\s([(\-*\d+.\d+)\s]+)',mrifid)[ind_fid].split()[0:3]]
      right_ear=[float(ii) for ii in re.findall('REC\sSession\s[\d+|"(null)"\s]+[\d+|"(null)"]\s([(\-*\d+.\d+)\s]+)',mrifid)[ind_fid].split()[0:3]]
      left_ear=[float(ii) for ii in re.findall('LEC\sSession\s[\d+|"(null)"\s]+[\d+|"(null)"]\s([(\-*\d+.\d+)\s]+)',mrifid)[ind_fid].split()[0:3]]

      meg_filename='filt/%s.%s.ica.%.2f-%.2fHz_raw.fif'%(ss,bb,args.lo,args.hi)

   except:
      # beta tACS format
      try:
         mrifid = open('neuronav/%s/%s_bs.txt'%(ss,ss)).read()
      except:
         continue     
 
      nasion=[float(ii) for ii in re.findall('NEC\sEEG\sSession\s[\d+|"(null)"\s]+\s([(\-*\d+.\d+)\s]+)',mrifid)[0].split()[0:3]]
      right_ear=[float(ii) for ii in re.findall('REC\sEEG\sSession\s[\d+|"(null)"\s]+\s([(\-*\d+.\d+)\s]+)',mrifid)[0].split()[0:3]]
      left_ear=[float(ii) for ii in re.findall('LEC\sEEG\sSession\s[\d+|"(null)"\s]+\s([(\-*\d+.\d+)\s]+)',mrifid)[0].split()[0:3]]

      # This matches the MEG file
      meg_filename='filt/%s.%s.ica.%.2f-%.2fHz_raw.fif'%(ss,bb,args.lo,args.hi)

   # Get center of RAS needed to correct Brainsight RAS to Freesurfer RAS
   cras=sp.check_output(['mri_info','--cras', 'mri/%s_T1.nii'%ss],stderr=sp.STDOUT)
   cras = [float(ii) for ii in cras.split()]
   elp = np.vstack((nasion,left_ear,right_ear))-cras

   try:

      #if os.path.exists('coreg/%s.%s.coreg.ica.%.2f-%.2fHz_raw.fif'%(ss,bb,args.lo,args.hi)):
      #   continue


      # Find coil HPI info from FIF file corresponding to such fiducials (HPI)
      print 'Finding coil HPI'
      raw=mne.io.read_raw_fif(meg_filename, preload=True)
      hpi = raw.info['hpi_results']
      hpi=np.vstack([ii['r'] for ii in hpi[0]['dig_points']])

      print(hpi)

      print 'Setting montage'
      # Set the montage and save coregistered file
      d=mne.channels.read_dig_montage(point_names=['nasion', 'lpa', 'rpa'],hpi=hpi,
      elp=elp,transform=False,dev_head_t=True,unit='mm')
      raw.set_montage(d)
      raw.save('coreg/%s.%s.coreg.ica.%.2f-%.2fHz_raw.fif'%(ss,bb,args.lo,args.hi),overwrite=False)

      print 'Visualizing'
      # Visualize it
      if args.viz:
         mne.viz.plot_alignment(info=raw.info, subjects_dir=os.environ['SUBJECTS_DIR'],
         coord_frame='head',subject=ss,dig=True,meg=['sensors'],eeg=[],surfaces=['head'])
         mlab.view(180, 180)
         mlab.show()
   except:
      print 'Cannot process coreg of subj %s'%ss

      #mne.viz.plot_alignment(info=raw.info, subjects_dir=os.environ['SUBJECTS_DIR'],
      # coord_frame='head',subject=ss,dig=True,meg=['sensors'],eeg=[],surfaces=['head'])
      #mlab.view(180, 180)
      #mlab.show()

