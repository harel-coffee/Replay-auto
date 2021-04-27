import numpy as np, itertools, argparse, dill, pdb, re, os
import matplotlib.pyplot as plt
from scipy.stats import sem, ttest_ind, percentileofscore, t
from scipy.spatial.distance import hamming
from scipy.io import savemat

def plot_probs(avg_prob, sem_prob, noticks=True):

   colors=['r','g','b','w']

   if noticks:
      plt.gca().set_xticks([])
      plt.gca().set_yticks([])

   try:
      for ii,jj,kk in zip(avg_prob.T,sem_prob.T,range(5)):
         if not np.any(np.isnan(sem_prob)):
            plt.fill_between(range(5),ii-jj,ii+jj,facecolor=colors[kk],alpha=0.5,label='Key %d'%(kk+1))
            plt.xticks(range(5),['t=%d'%(ll+1) for ll in range(5)],fontsize=14)
         else:
            colors=['r','g','b','gray']
            plt.gca().set_color_cycle(colors)
            plt.plot(range(5), avg_prob,alpha=0.5)

         plt.legend()
         plt.xlabel('Ordinal position',fontsize=20)
         plt.ylabel('Key probability (95% CI)',fontsize=20)

   except:
      pass



def calc_transitions(type_dist, target, perms):

   if type_dist==1:
     return(np.array([hamming(ii,target) for ii in perms]))

   bins=[tuple(target[ii:ii+type_dist]) for ii in range(len(target)-type_dist+1)]

   dist=[]
   for jj in perms:
      pairs_jj=[tuple(jj[ii:ii+type_dist]) for ii in range(len(target)-type_dist+1)]
      dist.append(len(set(pairs_jj).intersection(bins)))

   return np.array(dist)



def perm_test(probs, p_true):
   
   perms=list(itertools.product([1,2,3,4],repeat=5))
   pvalue=np.zeros((len(prob),))

   for ind, ii in enumerate(prob):

      p_seq_null=np.zeros((len(perms),))

      for ind_jj,jj in enumerate(perms):

         p0 =prob[ind][0][jj[0]-1]
         p1 =prob[ind][1][jj[1]-1]
         p2 =prob[ind][2][jj[2]-1]
         p3 =prob[ind][3][jj[3]-1]
         p4 =prob[ind][4][jj[4]-1]
         p_seq_null[ind_jj] = p0*p1*p2*p3*p4

      pvalue[ind]=percentileofscore(p_seq_null, p_true[ind],'strict')

   return pvalue

if __name__=='__main__':

   # Create a default list with any subject code in the meg.txt file
   all_subjs=re.findall('([A-Z]+)\.day\d\/.+\n',open('meg.txt').read())
   all_subjs=list(set(all_subjs))

   parser = argparse.ArgumentParser(description='Calculate decoder results')
   parser.add_argument('--block',dest='block')
   args=parser.parse_args()

   criteria=['proba_pred_bal_accuracy','f1_macro','proba_pred_weighted_accuracy']
   criteria_str=['Bal.acc.','F1','Wei.acc.']

   all_rates=np.empty((len(all_subjs),len(criteria)))
   all_rates[:]=np.nan
   all_pseqs=np.empty((len(all_subjs),len(criteria)))
   all_pseqs[:]=np.nan

   seq=[4,1,3,2,4]
   for ind_cc,cc in enumerate(criteria):

      if not os.path.exists('kp_decoder_results/results_%s_%s.pkl'%(args.block,cc)):
      #if 1==1:
         rates=[]
         preds=[]
         probs=[]
         pseqs=[]

         for ss in all_subjs:
 
            ypred_all=[]

            #filename='kp_decoder_broad/%s.cv.%s.%s.pkl'%(ss,args.block,cc)
            filename='kp_decoder_broad/%s.cv.day1.%s.pkl'%(ss,cc)
            #filename='backup/kp_decoder_broad/%s.cv.day1.%s.pkl'%(ss,cc)

            try:
               out=dill.load(open(filename))
               prob=out[-1]
            except:
               print ss
               rates.append(np.nan)
               preds.append([])
               probs.append([])
               pseqs.append([])
               continue            

            p_seq   =np.zeros((len(prob),))
            detected =np.empty((len(prob),))
            detected[:] =np.nan

            for ind, ii in enumerate(prob):
               
               ypred=[np.argmax(kk)+1 for kk in ii]
               ypred_all.append(ypred)

               print '%d of %d' %(ind,len(prob))
               p0 =prob[ind][0][seq[0]-1]
               p1 =prob[ind][1][seq[1]-1]
               p2 =prob[ind][2][seq[2]-1]
               p3 =prob[ind][3][seq[3]-1]
               p4 =prob[ind][4][seq[4]-1]
               
               p_seq[ind] = p0*p1*p2*p3*p4
            
            ypred_all=np.array(ypred_all)

            inds = perm_test(prob,p_seq) >= 95
            rate=len(ypred_all[inds])/float(len(ypred_all))            
        
            if rate > 0:
               rates.append(rate)
               pseqs.append(p_seq[inds])
               preds.append(ypred_all[inds])
               probs.append([kk for kk,jj in zip(prob,inds) if jj])
            else:
               print ss
               pdb.set_trace()
               rates.append(0)
               preds.append([])
               probs.append([])
               pseqs.append([])

         results=[]
         print len(all_subjs), len(rates), len(preds), len(probs)
         for ii,jj,kk,ll,mm in zip(all_subjs,rates,preds,probs,pseqs): 
            print ii
            results.append((ii,jj,kk,ll,mm)) 
	      
         dill.dump(results, open('kp_decoder_results/results_%s_%s.pkl'%(args.block,cc),'wb'))

      else:
         #results=dill.load(open('backup/kp_decoder_results/results_%s_%s.pkl'%(args.block,cc)))
         results=dill.load(open('kp_decoder_results/results_%s_%s.pkl'%(args.block,cc)))

      for ind_gg,_ in enumerate(results):
         all_rates[ind_gg][ind_cc] = results[ind_gg][1]  
         all_pseqs[ind_gg][ind_cc] = np.mean(results[ind_gg][-1])

   if not np.all(all_rates==np.nan):

      # Find best rates and models for each subject and save list of models
      with open('kp_decoder_broad/%s_models.txt'%args.block,'wt') as file:

         print all_rates

         best_inds=[np.nanargmax(ii) if np.any(ii>0) else np.nan for ii in all_rates]

         out_subjs=[]
         cv_cnf_mat=[]
         train_cnf_mat=[]
         pval_cv=[]
         subj_probs=[]
         top_rates=[]
         for ind_gg,_ in enumerate(results):
            if not np.isnan(best_inds[ind_gg]):
               #print all_rates[ind_gg][best_inds[ind_gg]],all_pseqs[ind_gg][best_inds[ind_gg]],
               #criteria[best_inds[ind_gg]]
               filename='kp_decoder_broad/%s.cv.%s.%s.pkl'%(all_subjs[ind_gg],
               args.block,criteria[best_inds[ind_gg]])
               #filename='backup/kp_decoder_broad/%s.cv.%s.%s.pkl'%(all_subjs[ind_gg],
               #args.block,criteria[best_inds[ind_gg]])
               file.write('%.4f, %.6f, %s, %s\n'%(all_rates[ind_gg][best_inds[ind_gg]],
               all_pseqs[ind_gg][best_inds[ind_gg]],
               all_subjs[ind_gg],filename))

               results=dill.load(open('kp_decoder_results/results_%s_%s.pkl'%(args.block,\
               criteria[best_inds[ind_gg]])))
               subj_probs.append(np.mean(results[ind_gg][3],0))

               top_rates.append(all_rates[ind_gg][best_inds[ind_gg]])
               with open(filename) as pkl_file:
                  out = dill.load(pkl_file)               
                  cv_cnf_mat.append(out[6])
                  train_cnf_mat.append(out[7])
                  pval_cv.append(out[5])

                  #if pval_cv[ind_gg] > 0.05:
                  #   pdb.set_trace()
                  
                  out_subjs.append(all_subjs[ind_gg])

            else:
               print all_subjs[ind_gg],all_rates[ind_gg]

         #savemat('kp_decoder_broad/kp_decoder_eval',
         #{'cv_cnf_mat':cv_cnf_mat,'train_cnf_mat':train_cnf_mat,'pval_cv':pval_cv,'out_subjs':out_subjs})
         
         subj_probs = [ ii for ii,jj in zip(subj_probs,pval_cv) if jj < 0.05 ]
         top_rates = [ ii for ii,jj in zip(top_rates,pval_cv) if jj < 0.05 ]

         pdb.set_trace()

         avg_prob = np.mean(subj_probs,0)
         #sem_prob = sem(subj_probs,0,nan_policy='omit')
         h_delta=t.ppf(1-(0.05/2),30-1)*np.nanstd(subj_probs,0)/np.sqrt(30)

         # Group plot
         plt.figure(figsize=(8,8))
         plt.gcf().patch.set_facecolor('w')

         #plot_probs(avg_prob, sem_prob, False)
         plot_probs(avg_prob, h_delta, False)

         plt.gcf().savefig('play_kp_probs.pdf')

         # Histogram of detection rates
         plt.figure(figsize=(10,10))
         plt.gcf().patch.set_facecolor('w')
	
         plt.hist(top_rates,5,facecolor='gray',alpha=0.75,histtype="stepfilled", 
         edgecolor='black', linewidth=1.2)

         xt=plt.gca().get_xticks()
         plt.gca().set_xticklabels(['%d%%'%(100*ii) if ind >0 and ind <len(xt)-1 else ''  
         for ind,ii in enumerate(xt)])

         plt.setp(plt.gca().get_xticklabels(),fontsize=18)
         plt.ylim([0,12])

         plt.xlabel('Detection rate',fontsize=30)
         plt.ylabel('#Subjects',fontsize=30)

         plt.gcf().savefig('seq_det_rates_train.pdf')
         plt.show()
