# The sip import and sets below address a version inconsistency problem 
# of mayavi and PyQt
'''
import sip
sip.setapi('QDate', 2)
sip.setapi('QDateTime', 2)
sip.setapi('QString', 2)
sip.setapi('QTextStream', 2)
sip.setapi('QTime', 2)
sip.setapi('QUrl', 2)
sip.setapi('QVariant', 2)
from mayavi import mlab
from surfer import Brain
'''

import mne, numpy as np, matplotlib.pyplot as plt, os.path
import argparse, os, re, itertools,pdb

if __name__ == '__main__':

   # Read all subject codes from meg.txt
   all_subjs=re.findall('([A-Z]+)\.day\d\/.+\n',open('meg.txt').read())
   all_subjs=list(set(all_subjs))
 
   # Parse input arguments
   parser = argparse.ArgumentParser(description='FWD model.')
   parser.add_argument('--meg_codes',dest='subjs',default=all_subjs,nargs='+')
   parser.add_argument('--blocks',dest='blocks',nargs='+')
   parser.add_argument('--lo',dest='lo',type=float,default=1)
   parser.add_argument('--hi',dest='hi',type=float,default=100)
   parser.add_argument('--njobs', dest='njobs',type=int,default=1)
   parser.add_argument('--viz',action='store_true')
   args=parser.parse_args()

   for ss,bb in itertools.product(args.subjs, args.blocks):

      # Load bem solution
      bem_folder='%s/%s/bem'%(os.environ['SUBJECTS_DIR'],ss)

      if os.path.isfile('%s/%s_fwd.%s.fif'%(bem_folder,ss,bb)):
         print ss,bb
         continue

      src=mne.read_source_spaces('%s/%s-oct6-src.fif'%(bem_folder,ss))

      # Read the coreg file, with HEAD coordinates in MRI coodinates
      try:
         raw=mne.io.read_raw_fif('coreg/%s.%s.coreg.ica.%.2f-%.2fHz_raw.fif'\
         %(ss,bb,args.lo,args.hi), preload=True)
      except:
         print 'Cannot process MEG file coreg/%s.%s.coreg.ica.%.2f-%.2fHz_raw.fif'%(ss,bb,args.lo,args.hi)
         continue

      # Make the forward
      
      bem_file='%s/%s-8196-bem-sol.fif'%(bem_folder,ss)
      bem_sol = mne.read_bem_solution(bem_file)
      fwd = mne.make_forward_solution(info=raw.info, src=src, bem=bem_sol,\
      trans=None, n_jobs=args.njobs, meg=True, eeg=False, ignore_ref=False) 
      mne.write_forward_solution('%s/%s_fwd.%s.fif'%(bem_folder,ss,bb), fwd, overwrite=True)

      # This visually projects the forward source space (in HEAD coords) onto the surfaces 
      # in (MRI coords). If properly coregistered, HEAD will be in MRI coordinates 
      # and all will work
      #brain = Brain(ss, 'lh', 'inflated', subjects_dir=os.environ['SUBJECTS_DIR'])
      #surf = brain.geo['lh']

      #for ii in range(len(fwd['src'])):
      #   vertidx = np.where(fwd['src'][ii]['inuse'])[0]
      #   mlab.points3d(surf.x[vertidx], surf.y[vertidx],surf.z[vertidx], 
      #   color=(1, 1, 0), scale_factor=1.5)
      #mlab.show()
