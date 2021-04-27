import mne, sys, os, argparse, re


if __name__ == '__main__':

   # Read all subject codes from meg.txt
   all_subjs=re.findall('([A-Z]+)\.day\d\/.+\n',open('meg.txt').read())
   all_subjs=list(set(all_subjs))

   # Parse input arguments
   parser = argparse.ArgumentParser(description='BEM model.')
   parser.add_argument('--meg_codes',dest='subjs',default=all_subjs,nargs='+')
   args=parser.parse_args()

   lut_file='/data/claudinolm/nrsa/seq_learning/freesurfer/subjects/BN_Atlas_246_LUT.txt'

   for ss in args.subjs:
   
      bem_folder='%s/%s/bem'%(os.environ['SUBJECTS_DIR'],ss)

      # Make the whole brain model
      model = mne.make_bem_model(ss,ico=None,conductivity=[0.3])
      mne.write_bem_surfaces('%s/%s-8196-bem.fif'%(bem_folder,ss), model)
      
      bem_file='%s/%s-8196-bem-sol.fif'%(bem_folder,ss)
      if not os.path.exists(bem_file):
         bem_sol = mne.make_bem_solution(model)
         mne.write_bem_solution(bem_file, bem_sol)

      src_file='%s/%s-oct6-src.fif'%(bem_folder,ss)
      if not os.path.exists(src_file):
         # Setup surface source space
         src = mne.setup_source_space(ss, spacing='oct6',n_jobs=1)

         # Add hippocampus to source
         hipp_labels = [u'rHipp_L', u'rHipp_R', u'cHipp_L', u'cHipp_R']

         vol_src = mne.setup_volume_source_space(subject=ss, pos=5.0, 
         mri='BN_Atlas_subcortex.mgz',lut_path=lut_file,bem=bem_file,
         volume_label=hipp_labels,verbose=True)

         src+=vol_src

         mne.write_source_spaces(src_file, src, overwrite=True)


