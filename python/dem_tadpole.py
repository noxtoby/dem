#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,sys,time,glob
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
import pystan

#* Plotting
from matplotlib import pyplot as plt
# import seaborn as sn
# colors = sn.color_palette('viridis', 9)
#colors = sn.color_palette('colorblind', 5)
plt.style.use('ggplot')
from matplotlib import rcParams # For changing default figure size
plt.rcParams.update({'font.size': 12})

# #* KDE EBM
# from kde_ebm.mixture_model import fit_all_kde_models, get_prob_mat
# from kde_ebm.plotting import mixture_model_grid, mcmc_uncert_mat, mcmc_trace, stage_histogram
# from kde_ebm.mcmc import mcmc, parallel_bootstrap, bootstrap_ebm, bootstrap_ebm_fixedMM, bootstrap_ebm_return_mixtures
#* EBM Utilities
import dem_utilities as ute

#* Saving
import pickle


#*******************************************************
rundate = str(datetime.now().date()).replace('-','')
nom = 'dem_tadpole'
fname_save = "{0}-results-{1}-pickle".format(nom,rundate)
#* Preliminaries
daysInAYear = 365.25

#*** Prep: data loading
fname = '{0}-prepped.csv'.format(nom)
df, dem_markers = ute.preliminaries(fname_save=fname)
dx_column = 'DX'
#* Change to percentage of ICV
for m in ['WholeBrain','Hippocampus','Ventricles','Entorhinal']:
    df[m] = 100*df[m]

#*** DEM 1. Fit individual gradients
fname_gradients = '{0}-prepped-differential.csv'.format(nom)
df_dem = ute.dem_gradients(df=df,markers=dem_markers,fname_save=fname_gradients)
#* Postselect: exclude non-progressing, non-abnormal
df_dem_, df_postselection = ute.dem_postselect(df_dem=df_dem,markers=dem_markers,dx_col='DX.bl')

#* FIXME: this is temporary - subsample to reduce the dimensions (fitting takes too long!)
#- Sort biomarkers to approximate getting even coverage
df_dem__ = df_dem_.sort_values(by=[d+'-mean' for d in dem_markers])
df_dem__.reset_index(drop=True, inplace=True)
n_subsample = df_dem__.shape[0]
s = np.arange(0,df_dem__.shape[0],int(df_dem__.shape[0]/n_subsample))
df_dem__ = df_dem__.loc[s]

#*** DEM 2.1 Fit Gaussian Process regression 
betancourt = True
if betancourt:
    dem_stan_code_gpr = os.path.join('/Users/noxtoby/Code/GitHub/DPM_benchmark/results_Neil','gp_betancourt.stan')
else:
    dem_stan_code_gpr = os.path.join('/Users/noxtoby/Code/GitHub/DPM_benchmark/results_Neil','gp.stan')
    # dem_stan_code_gpr = os.path.join('/Users/noxtoby/Documents/Cluster/stan/projects/DEM','gpfit.stan')
fname_save_pystan = "{0}-results-{1}-pickle-pystan".format(nom,rundate)
check_flag = ute.check_for_save_file(file_name=fname_save_pystan,function=None)
if check_flag == 1:
    print('dem_adni():          Loading existing StanModel for Gaussian Process DEM')
    pickle_file = open(fname_save_pystan,'rb')
    pystan_results = pickle.load(pickle_file)
    dem_gpr_stan_model = pystan_results["dem_gpr_stan_model"]
    pickle_file.close()
else:
    print('dem_adni():          Compiling StanModel for Gaussian Process DEM')
    # dem_linreg_stan = pystan.StanModel(file='linreg.stan',model_name='linreg')
    # dem_linreg_robust_stan = pystan.StanModel(file='linreg_robust.stan',model_name='linreg_robust') # Student's t, instead of Gaussian
    dem_gpr_stan_model = pystan.StanModel(file=dem_stan_code_gpr,model_name='dem_gpr')
    #* Save
    pystan_results = { "dem_gpr_stan_model": dem_gpr_stan_model } # "dem_linreg_stan_model": dem_linreg_stan, "dem_linreg_robust_stan_model": dem_linreg_robust_stan
    pickle_file = open(fname_save_pystan,'wb')
    pickle_output = pickle.dump(pystan_results,pickle_file)
    pickle_file.close()


#* Fit the DEMs
fname_save_pystan_fits = "{0}-results-{1}-pickle-pystan-fits".format(nom,rundate)
check_flag = ute.check_for_save_file(file_name=fname_save_pystan_fits,function=None)

if check_flag: #"dem_gpr_stan_model_fits" in pystan_results:
    print('fit_dem():           Not fitting DEms: existing results detected.')
    pickle_file = open(fname_save_pystan_fits,'rb')
    pystan_results = pickle.load(pickle_file)
    dem_gpr_stan_model_fits = pystan_results["dem_gpr_stan_model_fits"]
    pickle_file.close()
    #dem_gpr_stan_model_fits = pystan_results["dem_gpr_stan_model_fits"]
else:
    print('fit_dem():           Fitting DEms using GP regression')
    dem_gpr_stan_model_fits = ute.fit_dem(df_dem=df_dem__,markers=dem_markers,stan_model=dem_gpr_stan_model,betancourt=betancourt)
    
    #* Save the fits to a new pickle file - the StanModel mmust be unpickled first
    pystan_results["dem_gpr_stan_model_fits"] = dem_gpr_stan_model_fits
    pickle_file = open(fname_save_pystan_fits,'wb')
    pickle_output = pickle.dump(pystan_results,pickle_file)
    pickle_file.close()


#* Next: Diagnostics (Plot etc.: ), 
# https://github.com/betanalpha/jupyter_case_studies/blob/master/principled_bayesian_workflow/principled_bayesian_workflow.ipynb

#* Then: Integrate to get trajectories, Stage individuals (per trajectory and across all)
#* Later: consider Cross-validation

for m in dem_markers: #['Ventricles']:
    x = df_dem__[m+'-mean']
    y = df_dem__[m+'-grad']
    #x_predict = df_dem_fits.loc[df_dem_fits.Marker==m,'x_predict'].values[0]
    x_predict = np.linspace(np.nanmin(x),np.nanmax(x),20)
    
    fit = dem_gpr_stan_model_fits.loc[dem_gpr_stan_model_fits.Marker==m,'pystan_fit_gpr'].values[0]
    print(fit)
    
    samples = fit.extract(permuted=True, inc_warmup=False)
    rho = np.median(samples['rho'],0)
    alpha = np.median(samples['alpha'],0)
    sigma = np.median(samples['sigma'],0)
    f_predict = np.median(samples['f_predict'],0) # GP posterior mean
    y_predict = np.median(samples['y_predict'],0) # GP posterior spread
    
    lin = df_postselection.loc[df_postselection.Marker==m,'Normal-Threshold'].values*[1,1]
    
    fig,ax = plt.subplots()
    ax.plot(x_predict,samples['f_predict'].T,'w')
    ax.plot(df_dem_[m+'-mean'],df_dem_[m+'-grad'],'b.',label=m+' (all)')
    ax.plot(x,y,'k.',label=m+' (sampled)')
    ax.plot(lin,ax.get_ylim(),label='CN median')
    ax.plot(x_predict,f_predict,'r',label='f predict')
    ax.legend()
    fig.show()







#*** DEM 2.2 Cross-validation: 10-fold
#  NOTE: Could use CV to do event postselection (remove events having multimodal positional variance)
if "mixtures_cv" in ebm_results:
    print('ebm_2_cv():           Not running CV: existing results detected.')
    kde_mixtures_cv = ebm_results["mixtures_cv"]
    mcmc_samples_cv = ebm_results["mcmc_samples_cv"]
    seqs_cv = ebm_results["sequences_cv"]
else:
    print('ebm_2_cv():           Running 10-fold cross-validation')
    kde_mixtures_cv, mcmc_samples_cv, seqs_cv = ebm_ute.ebm_2_cv(x=x,y=y,events=events)
    #* Save
    ebm_results["mixtures_cv"] = kde_mixtures_cv
    ebm_results["mcmc_samples_cv"] = mcmc_samples_cv
    ebm_results["sequences_cv"] = seqs_cv
    pickle_file = open(fname_save,'wb')
    pickle_output = pickle.dump(ebm_results,pickle_file)
    pickle_file.close()


#* FIXME: Cross-validation similarity: Hellinger distance between positional variance diagrams (vectorised)
cvs = ebm_ute.cv_similarity(mcmc_samples_cv,seq=seq_ml)


#*** EBM 2.3 Cross-validation: Bootstrapping
bs_n = 100 # Number of bootstrap (re)samples
bs_mcmc = len(mcmc_samples) # MCMC iterations

#*** EBM 2.3.1 Bootstrapping with fixed mixture models
fixedMM_flag = True
if "mixtures_bs_fixedMM" in ebm_results:
    print('ebm_2_bs():           Not running bootstrap CV: existing results detected.')
    kde_mixtures_bs_fixedMM = ebm_results["mixtures_bs_fixedMM"]
    mcmc_samples_bs_fixedMM = ebm_results["mcmc_samples_bs_fixedMM"]
    seqs_bs_fixedMM = ebm_results["sequences_bs_fixedMM"]
else:
    bs_text = 'ebm_2_bs():           Running bootstrap cross-validation '
    bs_text += '(n = {0}; n_mcmc = {1}) - '.format(bs_n,bs_mcmc) + fixedMM_flag*'not ' + 'refitting mixture models'
    print(bs_text)
    kde_mixtures_bs_fixedMM, mcmc_samples_bs_fixedMM, seqs_bs_fixedMM = ebm_ute.ebm_2_bs(
        x=x,
        y=y,
        events=events,
        fixed_mixture_models_flag=fixedMM_flag,
        mixtures=kde_mixtures,
        n_bs=bs_n,
        n_mcmc=bs_mcmc
        )
    #* Save
    ebm_results["mixtures_bs_fixedMM"] = kde_mixtures_bs_fixedMM
    ebm_results["mcmc_samples_bs_fixedMM"] = mcmc_samples_bs_fixedMM
    ebm_results["sequences_bs_fixedMM"] = seqs_bs_fixedMM
    pickle_file = open(fname_save,'wb')
    pickle_output = pickle.dump(ebm_results,pickle_file)
    pickle_file.close()

#*** EBM 2.3.2 Bootstrapping with refit mixture models
fixedMM_flag = False
if "mixtures_bs" in ebm_results:
    print('ebm_2_bs():           Not running bootstrap CV: existing results detected.')
    kde_mixtures_bs = ebm_results["mixtures_bs"]
    mcmc_samples_bs = ebm_results["mcmc_samples_bs"]
    seqs_bs = ebm_results["sequences_bs"]
else:
    bs_text = 'ebm_2_bs():           Running bootstrap cross-validation '
    bs_text += '(n = {0}; n_mcmc = {1}) - '.format(bs_n,bs_mcmc) + fixedMM_flag*'not ' + 'refitting mixture models'
    print(bs_text)
    kde_mixtures_bs, mcmc_samples_bs, seqs_bs = ebm_ute.ebm_2_bs(
        x=x,
        y=y,
        events=events,
        fixed_mixture_models_flag=fixedMM_flag,
        mixtures=kde_mixtures,
        n_bs=bs_n,
        n_mcmc=bs_mcmc
        )
    #* Save
    ebm_results["mixtures_bs"] = kde_mixtures_bs
    ebm_results["mcmc_samples_bs"] = mcmc_samples_bs
    ebm_results["sequences_bs"] = seqs_bs
    pickle_file = open(fname_save,'wb')
    pickle_output = pickle.dump(ebm_results,pickle_file)
    pickle_file.close()


#*** Positional Variance Diagrams
pvd_ml, seq_ml = ebm_ute.extract_pvd(ml_order=seq_ml,samples=mcmc_samples)
pvd_cv, seq_cv = ebm_ute.extract_pvd(ml_order=seqs_cv,samples=mcmc_samples_cv)
pvd_bs_fixedMM, seq_bs_fixedMM = ebm_ute.extract_pvd(ml_order=seqs_bs_fixedMM,samples=mcmc_samples_bs_fixedMM)
pvd_bs, seq_bs = ebm_ute.extract_pvd(ml_order=seqs_bs,samples=mcmc_samples_bs)

#*** Positional Variance, reordered to match labels in events
reorder_ml = np.argsort(seq_ml)
reorder_cv = np.argsort(seq_cv)
reorder_bs_fixedMM = np.argsort(seq_bs_fixedMM)
reorder_bs = np.argsort(seq_bs)
pvd_ml_ = pvd_ml[:][reorder_ml]
pvd_cv_ = pvd_cv[:][reorder_cv]
pvd_bs_fixedMM_ = pvd_bs_fixedMM[:][reorder_bs_fixedMM]
pvd_bs_ = pvd_bs[:][reorder_bs]


seq_ = seq_bs
pvd_ = pvd_bs_
n_biomarkers = pvd_.shape[0]

#* Check for "strong non-unimodality" in each row of the PVD => subsequently remove events
# 1. Fit a KDE with bandwidth of one event
# 2. Calculate the first & second derivatives to find the roots => peaks/modes
# 3. Assess the separation of the two highest peaks and the relative magnitude
#    Criteria for removal: 
#    - If peak magnitude is within a factor of two (larger peak is less than 2*)
#      and
#    - If separated by greater than 3 events

import sklearn as skl
kde_bw = 1 # bandwidth in "events"
kde = skl.neighbors.KernelDensity(kernel='gaussian',bandwidth=kde_bw)

pdfz = []
remove = []
x_ = np.linspace(0,n_biomarkers,200).reshape(-1,1)
for k in seq_:
    mkr = events[k].replace('-detrended','')
    if mkr[0:3]=='EMQ':
        mkr = 'EMQ'
    ex = pvd_[:][k].reshape(-1,1)
    ex = ( ex/np.max(ex) ) * 5e3 # Rescale for numerical convenience when fitting KDE
    ex = ex.astype(int)
    #* Regenerate samples for KDE fitting
    s = []
    for e in range(len(ex)):
        s.append(e*np.ones(int(ex[e])))
    s = np.concatenate(s).reshape(-1,1)
    
    f = kde.fit(X=s)
    pdf = np.exp(f.score_samples(x_))
    
    dpdx = np.gradient(pdf,n_biomarkers/(len(x_)-1))
    d2pdx2 = np.gradient(dpdx,n_biomarkers/(len(x_)-1))
    d3pdx3 = np.gradient(d2pdx2,n_biomarkers/(len(x_)-1))
    #* Identify zeros (where dpdx sign switches)
    zeroz = np.where(np.diff(np.sign(dpdx))!=0)[0] + 1
    if len(zeroz)!=0:
        #* Identify peaks (where d2pdx2 is negative)
        peakz = zeroz[ d2pdx2[zeroz]<0 ]
        #* Calculate separation and ratio
        ratio = pdf[peakz]/np.max(pdf[peakz])
        separation = np.diff(x_[peakz].flatten())
        if np.any( separation>3 ):
            if any( (ratio[ratio>0.5] - 1) < 0 ):
                remove.append(k)
        fig,ax=plt.subplots(3,1,figsize=(9,6))
        ax[0].plot(x_,pdf/np.max(pdf))
        ax[0].set_title('{0} positional variance'.format(mkr))
        ax[0].set_ylabel('KDE')
        ax[1].plot(x_,dpdx,'-',x_[zeroz-1],dpdx[zeroz-1],'o')
        ax[1].set_ylabel(r'$\nabla_x$(KDE)')
        ax[2].plot(x_,d2pdx2,'-',x_[zeroz-1],d2pdx2[zeroz-1],'o')
        ax[2].set_ylabel(r'$\nabla^2_x$(KDE)')
        fig.show()
        #fig.savefig('{0}-multimodal_FBS-{1}.png'.format(nom,mkr))
    
    #pdfz.append(pdf)

print('You need to remove these multimodal events:')
print(',\n'.join([events[r] for r in remove]))

# fig,ax=plt.subplots()
# ax.fill(x_, pdfz[0], fc='#AAAAFF')
# fig.show()






############ It's no longer pretty from here, but it's easier to tweak the plots ############
seq_ = seq_bs
pvd_ = pvd_bs_

fig, ax = plt.subplots(1,2,figsize=(14, 5),sharey=False)
labels = events_labels
labels_ = [labels[i].replace('TOTAL','').replace('TOT','').replace('-detrended','') for i in seq_]
ax[0].imshow(pvd_ml_[:][seq_], interpolation='nearest', cmap='Oranges')
ax[0].set_title('Most Probable',fontsize=24); txt1 = 'ML'
# ax[1].imshow(pvd_cv_[:][seq_], interpolation='nearest', cmap='Oranges')
# ax[1].set_title('Cross-Validation',fontsize=28); txt2 = 'CV'
#ax[1].imshow(pvd_bs_fixedMM_[:][seq_], interpolation='nearest', cmap='Oranges')
#ax[1].set_title('Sequence Bootstrap',fontsize=28); txt2 = 'PBS'
ax[1].imshow(pvd_bs_[:][seq_], interpolation='nearest', cmap='Oranges')
ax[1].set_title('Full Bootstrap',fontsize=24); txt2 = 'FBS'

stp = 1
fs = 12
if n_biomarkers>8:
    stp = 3
    fs = 10
tick_marks_x = np.arange(0,n_biomarkers,stp)
x_labs = range(1, n_biomarkers+1,stp)
ax[0].set_xticks(tick_marks_x)
ax[1].set_xticks(tick_marks_x)
ax[0].set_xticklabels(x_labs, rotation=0,fontsize=fs)
ax[1].set_xticklabels(x_labs, rotation=0,fontsize=fs)
tick_marks_y = np.arange(n_biomarkers)
ax[0].set_yticks(tick_marks_y+0.5)
ax[1].tick_params(left='off',labelleft='off')
ax[0].tick_params(axis='y',color='w')
labels_trimmed = [x[2:].replace('_', ' ') if x.startswith('p_') else x.replace('_', ' ') for x in labels_]
ax[0].set_yticklabels(labels_trimmed,#,np.array(labels_trimmed, dtype='object')[seq_],
                   rotation=0, #ha='right',
                   rotation_mode='anchor',
                   fontsize=18)
ax[0].set_ylabel('Measure', fontsize=28)
ax[0].set_xlabel('Sequence', fontsize=28)
ax[1].set_xlabel('Sequence', fontsize=28)
ax[0].grid(False)
ax[1].grid(False)
fig.tight_layout()
fig.show()
f_name = '{0}-PVD_{1}vs{2}-{3}.png'.format(nom,txt1,txt2,rundate)
# fig.savefig(f_name,dpi=300)



# mcmc_samples_bs_ = []
# for k in range(100):
#      kk = np.arange(k*100,(k+1)*100)
#      mcmc_samples_bs_.append([mcmc_samples_bs[l] for l in kk])
#
# mcmc_samples_bs = mcmc_samples_bs_



#*** EBM 3. Longitudinal Consistency Analysis: for sufficient event data per individual, per visit
# - Staging consistency: an individual's followup data should progress (not regress) through the model stages
df_detrended.rename(columns={'EYO_x':'EYO'},inplace=True)
df_staging = df_detrended[['Subject_ID','Visit',dx_column,'Mutation','MutationType','Years_bl','EYO']+events].copy()
x_long = df_staging[events].values
df_staging['Fraction missing data'] = np.sum(np.isnan(x_long),axis=1)/x_long.shape[1]
stage_column = "Model stage"

#*** 3.1 Staging
#* Max Like
prob_mat_ml, stages_long_ml, stage_likelihoods_long_ml = ebm_ute.ebm_3_staging(x=x_long,mixtures=kde_mixtures,samples=mcmc_samples)
#* CV: 10-fold
prob_mat_cv, stages_long_cv, stage_likelihoods_long_cv = ebm_ute.ebm_3_staging(x=x_long,mixtures=kde_mixtures_cv,samples=mcmc_samples_cv)
#* CV: Bootstrap
prob_mat_bs, stages_long_bs, stage_likelihoods_long_bs = ebm_ute.ebm_3_staging(x=x_long,mixtures=kde_mixtures_bs,samples=mcmc_samples_bs)
# prob_mat_bs_fixedMM, stages_long_bs_fixedMM, stage_likelihoods_long_bs_fixedMM = ebm_ute.ebm_3_staging(x=x_long,mixtures=kde_mixtures_bs_fixedMM,samples=mcmc_samples_bs_fixedMM)

#*** Choose ML
prob_mat = np.array([p for p in prob_mat_ml])
stages_long = np.array([s for s in stages_long_ml])
stage_likelihoods_long = np.array([sl for sl in stage_likelihoods_long_ml])
prob_mat_mean = np.mean(prob_mat,axis=0)

#*** Choose CV
prob_mat = np.array([p for p in prob_mat_cv])
stages_long = np.array([s for s in stages_long_cv])
stage_likelihoods_long = np.array([sl for sl in stage_likelihoods_long_cv])
# Average over folds
prob_mat = np.mean(prob_mat,axis=0)
stages_long = np.round(np.mean(stages_long,axis=0)).astype(int)
stage_likelihoods_long = np.mean(stage_likelihoods_long,axis=0)
prob_mat_mean = np.mean(prob_mat,axis=0)

#*** Choose FBS
prob_mat = np.array([p for p in prob_mat_bs])
stages_long = np.array([s for s in stages_long_bs])
stage_likelihoods_long = np.array([sl for sl in stage_likelihoods_long_bs])
# Average over folds
prob_mat = np.mean(prob_mat,axis=0)
stages_long = np.round(np.mean(stages_long,axis=0)).astype(int)
stage_likelihoods_long = np.mean(stage_likelihoods_long,axis=0)
prob_mat_mean = np.mean(prob_mat,axis=0)


# stages_long_mean = np.mean(stages_long,axis=0)
# stage_likelihoods_long_mean = np.mean(stage_likelihoods_long,axis=0)
# prob_mat_std = np.std(prob_mat,axis=0)
# stages_long_std = np.std(stages_long,axis=0)
# stage_likelihoods_long_std = np.std(stage_likelihoods_long,axis=0)
df_staging[stage_column] = pd.Series(stages_long)

#* Staging for groups
y_dx = df_staging[dx_column].values # longitudinal
stages_array = []
for k in np.unique(y_dx):
    stages_array.append(stages_long[(y_dx == k) & bl])
stages_array_labels = ['NC','MC']
fig, ax = ebm_ute.ebm_3_staging_plot(stages_array=stages_array,labels=stages_array_labels,normed=False)
f_name = '{0}-Staging-All_{1}_KDE.png'.format(nom,rundate)
fig.savefig(f_name,dpi=300)

# #* RISK OF UNBLINDING: Staging for mutation type AND diagnosis
# y_muttype = df_staging['MutationType'].map(lambda x: 1*(x=='PSEN1') + 2*(x=='APP')).values # longitudinal
# y_dx_muttype = [((dx==0)&(mt==1))*1 + ((dx==0)&(mt==2))*2 + ((dx==1)&(mt==1))*3 + ((dx==1)&(mt==2))*4 for dx,mt in zip(y_dx,y_muttype)]
# stages_array_dx_muttype = []
# for k in np.unique(y_dx_muttype):
#     stages_array_dx_muttype.append(stages_long[(y_dx_muttype == k) & bl])
# stages_array_dx_muttype_labels = ['NC-PSEN1','NC-APP','MC-PSEN1','MC-APP']
# fig, ax = ebm_ute.ebm_3_staging_plot(stages_array=stages_array_dx_muttype,labels=stages_array_dx_muttype_labels,normed=False)
# f_name = '{0}-Staging-All_{1}_KDE_dx-muttype.png'.format(nom,rundate)
# #fig.savefig(f_name,dpi=300)


# #* Staging for mutation type, regardless of diagnosis
# stages_array_muttype = []
# for k in np.unique(y_muttype):
#     stages_array_muttype.append(stages_long[(y_muttype == k) & bl])
# stages_array_muttype_labels = ['PSEN1','APP']
# fig, ax = ebm_ute.ebm_3_staging_plot(stages_array=stages_array_muttype,labels=stages_array_muttype_labels,normed=False)
# f_name = '{0}-Staging-All_{1}_KDE_muttype.png'.format(nom,rundate)


#* PSEN1 codon 200: 4th char of Mutation column is equal to 2
PSEN1_codon200 = [ x[-4]=='2' for x in df_staging['Mutation'].values]
df_staging['PSEN1 codon200'] = PSEN1_codon200
c200 = df_staging['Model stage'].values[(df_staging['PSEN1 codon200']==True) & bl]
c100 = df_staging['Model stage'].values[(df_staging['PSEN1 codon200']==False) & bl]
f,a=plt.subplots()
a.boxplot([
    c200,
    c100])
a.set_xticklabels(['PSEN Codon >200','PSEN Codon <200'])
a.set_ylabel('Model stage')
t,p = stats.ttest_ind(c200, c100, axis=0, equal_var=False)
# stats.mannwhitneyu(c200, c100, use_continuity=False)
a.set_title('t-test: t = {0} (p = {1})'.format(int(100*t)/100.0, int(100*p)/100.0))
f.show()


#*** 3.2 Consistency - WIP
# staging_consistency_results = ebm_ute.ebm_3_staging_consistency(df_staging=df_staging,model_pvd=pvd_cv,stage_column=stage_column)

print('Longitudinal consistency is not possible in this cohort\nbecause of too much missing data at followup')

#* FIXME: code this into ebm_ute.ebm_3_staging_consistency()
# For each individual having longitudinal data:
#   1- Calculate stages_i(t)
#   2- Soft Consistency: calculate whether these are within the PVD from CV
#   3- Hard consistency: calculate whether d(stages_i(t))/dt > 0
#   4- Medium consistency: calculate whether average gradient (linear fit) is positive stages_i = a_i*t + b_i: a_i > 0 ?
#df_staging['EBM stage likelihood'] = pd.Series(stage_likelihoods_long)

# missingness_threshold = 0.3
# df_temp = df_staging.loc[df_staging['Fraction missing data']> missingness_threshold].copy()
# cm = plt.cm.viridis(df_temp[dx_column])
# id_column = 'Subject_ID'
# u_ID = df_temp[id_column].unique()
# d = [] # d = np.zeros((len(u_ID),1))
# n_visits = []
# nonegative_diff = []
# nonegative_diff_avg = []
# empty_bool = []
# k = -1
# fig, ax = plt.subplots(1,1)
# for id in u_ID:
#     k += 1
#     rowz = id==df_temp[id_column]
#     xp = df_temp.loc[rowz,'Years_bl'].values
#     yp = df_temp.loc[rowz,'Model stage'].values
#     d.append(np.diff(yp)) # d[k] = np.diff(yp)
#     nonegative_diff.append(np.all(d[k]>=0))
#     nonegative_diff_avg.append(np.mean(d[k])>=0)
#     empty_bool.append(len(d[k])==0)
#     n_visits.append(len(d[k]))
#     ax.plot(xp,yp,'x-') #,c=cm[rowz])
# ax.set_ylabel('EBM stage')
# ax.set_xlabel('Years since baseline')
# fig.show()
# f_name = '{0}-Longitudinal_Consistency-{1}.png'.format(nom,rundate)
# fig.savefig(f_name,dpi=300)
#
# print('  Staging consistency results:')
# print('    - {0} individuals having longitudinal data ({1} visits total) with < {2}% missing event data'.format(
#     u_ID.shape[0]-sum(empty_bool),
#     df_temp.shape[0]-sum(empty_bool),
#     int(missingness_threshold*100)))
# print('    - {0} hard progressors (all non-negative staging changes)'.format(sum(nonegative_diff)))
# print('    - {0} medium progressors (non-negative stage changes on average)'.format(sum(nonegative_diff_avg)-sum(nonegative_diff)))
# print('FIXME    - ??? soft progressors (stage changes all within positional variance from CV)'.format())









#*** EBM vs EYO
xx = -df_staging.loc[bl,'EYO'].values
yy = df_staging.loc[bl,'Model stage'].values
MC = df_staging.loc[bl,'MC'].values == 1
fig = plt.figure(figsize=(6,4))
ax = plt.axes()
ax.plot(xx[~MC],yy[~MC],'bd',label='Noncarriers')
# plt.hold
ax.plot(xx[MC],yy[MC],'g^',label='Carriers')
ax.set_xlabel('EYO')
# ax.set_xticks([])
ax.set_ylabel('EBM stage')
ax.set_title('EBM vs EYO')
ax.axes.grid(False)
ax.set_xlim([np.min(ax.get_xlim()),0.5])
ax.plot([0,0],[0,n_biomarkers],label='EYO=0') # EYO = 0
ax.legend(loc='upper left',frameon=True)
fig.show()
f_name = '{0}-EBMvsEYO-{1}.png'.format(nom,rundate)
fig.savefig(f_name,dpi=300)








#*** Fit EBM-EYO curves
#* Max likelihood sequence
fname_save_pystan = "{0}-results-{1}-pickle-pystan".format(nom,rundate)
check_flag = ebm_ute.check_for_save_file(file_name=fname_save_pystan,function=None)
if check_flag == 1:
    print('eyo_ebm_curves():          Loading existing results')
    pickle_file = open(fname_save_pystan,'rb')
    pystan_results = pickle.load(pickle_file)
    model = pystan_results["model"]
    model_robust = pystan_results["model_robust"]
    model_GP = pystan_results["model_GP"]
    pickle_file.close()
else:
    print('eyo_ebm_curves():          Building EYO-EBM regression')
    model = pystan.StanModel(file='linreg.stan',model_name='linreg')
    model_robust = pystan.StanModel(file='linreg_robust.stan',model_name='linreg_robust') # Student's t, instead of Gaussian
    model_GP = pystan.StanModel(file='gp.stan',model_name='gp')
    #* Save
    pystan_results = { "model": model, "model_robust": model_robust, "model_GP": model_GP }
    pickle_file = open(fname_save_pystan,'wb')
    pickle_output = pickle.dump(pystan_results,pickle_file)
    pickle_file.close()

y_data = xx[MC]
y_lab = 'EYO'
x_data = yy[MC]
x_lab = 'Model stage'

#* SUPER BASIC linear regression
from scipy import stats, special
slope, intercept, r_value, p_value, std_err = stats.linregress(x_data,y_data)

#* Bayesian linreg
data = {'X':x_data, 'Y':y_data, 'N':len(x_data)}
fit = model.sampling(data=data) #, seed=42)
samples = fit.extract(permuted=True, inc_warmup=False)
a = np.median(samples['a'],0)
b = np.median(samples['b'],0)
#
# f,ax3 = plt.subplots(1,1,figsize=(10,5))
# ax3.plot(x_p,a*x_p + b,color='k',linewidth=2.0,linestyle='-',zorder=1,label='Bayesian linreg')
# ax3.plot(x_p,(a+da)*x_p + b,color='r',linewidth=2.0,linestyle='--',zorder=2,label='+/- std')
# ax3.plot(x_p,y_posterior_samples[:,1],color=(0.8,0.8,0.8),zorder=3,label='Post samples')
# ax3.plot(x_p,y_posterior_lower,color='r',linewidth=2.0,linestyle='--',zorder=4)
# ax3.plot(x_p,y_posterior_samples,color=(0.8,0.8,0.8),zorder=0)
# ax3.plot(x_data,y_data,color='b',marker='.',linestyle='',label='Data')
# ax3.set_yticks([])
# ax3.set_xlabel(x_lab)
# ax3.set_ylabel(y_lab)
# ax3.legend(frameon=True,fontsize=12)
#
# f.show()



#* Bayesian robust linreg
data_robust = {'X':x_data, 'Y':y_data, 'N':len(x_data), 'nu':(len(x_data)-1)}
fit_robust = model_robust.sampling(data=data_robust) #, seed=42)
samples_robust = fit_robust.extract(permuted=True, inc_warmup=False)
a_robust = np.median(samples_robust['a'],0)
b_robust = np.median(samples_robust['b'],0)

fig, ax = plt.subplots()
for k in range(len(samples_robust['a'])):
    if np.mod(k,40):
         ax.plot(EBM_stages,lm(samples_robust['a'][k],samples_robust['b'][k],EBM_stages),color=[0.9,0.9,0.9],zorder=0)
ax.plot(EBM_stages,lm(samples_robust['a'][k],samples_robust['b'][k],EBM_stages),color=[0.9,0.9,0.9],label='Posterior')
ax.plot(EBM_stages,lm(a_robust,b_robust,EBM_stages),'r',label='Mean',zorder=1)
ax.legend(loc='upper left')
ax.set_facecolor([1,1,1])
fig.show()

t_EBM_robust_samples = lm(samples_robust['a'],samples_robust['b'],EBM_stages)

t_EBM_robust_samples = np.ndarray(shape=(len(samples_robust['a']),len(EBM_stages)))
for k in range(t_EBM_robust_samples.shape[1]):
    t_EBM_robust_samples[:,k] = lm(samples_robust['a'],samples_robust['b'],EBM_stages[k])
t_EBM_robust = np.mean(t_EBM_robust_samples,axis=0) #lm(a_robust,b_robust,EBM_stages)
t_EBM_robust_iqr = stats.iqr(t_EBM_robust_samples,axis=0,rng=(25,75))
t_EBM_robust_upper = t_EBM_robust + t_EBM_robust_iqr/2
t_EBM_robust_lower = t_EBM_robust - t_EBM_robust_iqr/2

#* GP regression 
data_GP = {'x1':x_data, 'y1':y_data, 'N1':len(x_data), 
           'prior_std_inv_rho_sq': ( (np.nanmax(x_data)-np.nanmin(x_data))/1 )**2,
           'prior_std_eta_sq': ( (np.nanmax(y_data)-np.nanmin(y_data))/1 )**2,
           'prior_std_sigma_sq': ( (np.nanmax(y_data)-np.nanmin(y_data))/1 )**2/10}
fit_GP = model_GP.sampling(data=data_GP)
samples_GP = fit_GP.extract(permuted=True, inc_warmup=False)
rho_sq_GP = np.median(samples_GP['rho_sq'],0)
eta_sq_GP = np.median(samples_GP['eta_sq'],0)
sigma_sq_GP = np.median(samples_GP['sigma_sq'],0)
print(fit_GP)
x_p = np.linspace(0,n_biomarkers,101) # Interpolation points
y_posterior_samples, y_posterior_middle, y_posterior_upper, y_posterior_lower, RMSE = ebm_ute.evaluate_GP_posterior(x_p,x_data,y_data,rho_sq_GP,eta_sq_GP,sigma_sq_GP)

print('         LinReg       : EYO ~ {0}*EBM + {1}     (r = {2}; p = {3}; stderr = {4})'.format(np.around(slope,3),np.around(intercept,3),np.around(r_value,3),np.around(p_value,3),np.around(std_err,3)))
print('Bayesian linreg       : EYO ~ {0}*EBM + {1}'.format(np.around(a,3),np.around(b,3)))
print('Bayesian robust linreg: EYO ~ {0}*EBM + {1}'.format(np.around(a_robust,3),np.around(b_robust,3)))




#**** NC GPR
y_data_NC = xx[~MC]
x_data_NC = yy[~MC]
x_p_NC = np.linspace(0,np.max(x_data_NC),101) # Interpolation points

data_GP_NC = {'x1':x_data_NC, 'y1':y_data_NC, 'N1':len(x_data_NC), 
           'prior_std_inv_rho_sq': ( (np.nanmax(x_data_NC)-np.nanmin(x_data_NC))/1 )**2,
           'prior_std_eta_sq': ( (np.nanmax(y_data_NC)-np.nanmin(y_data_NC))/1 )**2,
           'prior_std_sigma_sq': ( (np.nanmax(y_data_NC)-np.nanmin(y_data_NC))/1 )**2/10}
fit_GP_NC = model_GP.sampling(data=data_GP_NC)
samples_GP_NC = fit_GP_NC.extract(permuted=True, inc_warmup=False)
rho_sq_GP_NC = np.median(samples_GP_NC['rho_sq'],0)
eta_sq_GP_NC = np.median(samples_GP_NC['eta_sq'],0)
sigma_sq_GP_NC = np.median(samples_GP_NC['sigma_sq'],0)
print(fit_GP_NC)
y_posterior_samples_NC, y_posterior_middle_NC, y_posterior_upper_NC, y_posterior_lower_NC, RMSE_NC = ebm_ute.evaluate_GP_posterior(x_p_NC,x_data_NC,y_data_NC,rho_sq_GP_NC,eta_sq_GP_NC,sigma_sq_GP_NC)

f,ax3 = plt.subplots(1,1,figsize=(10,5))
ax3.plot(x_p_NC,y_posterior_middle_NC,color='k',linewidth=2.0,linestyle='-',zorder=2,label='GP posterior mean. RMSE = {0}'.format(round(RMSE_NC,2)))
ax3.plot(x_p_NC,y_posterior_upper_NC,color='r',linewidth=2.0,linestyle='--',zorder=4,label='+/- std')
ax3.plot(x_p_NC,y_posterior_samples_NC[:,1],color=(0.9,0.9,0.9),zorder=6,label='Post samples')
ax3.plot(x_p_NC,y_posterior_lower_NC,color='r',linewidth=2.0,linestyle='--',zorder=8)
ax3.plot(x_p_NC,y_posterior_samples_NC,color=(0.8,0.8,0.8),zorder=0)
# ax3.plot(x_data_NC,y_data_NC,color='b',marker='.',linestyle='',label='Data')

ax3.plot(x_p,y_posterior_middle,color='b',linewidth=2.0,linestyle='-',zorder=3,label='GP posterior mean. RMSE = {0}'.format(round(RMSE,2)))
ax3.plot(x_p,y_posterior_upper,color='g',linewidth=2.0,linestyle='--',zorder=5,label='+/- std')
ax3.plot(x_p,y_posterior_samples[:,1],color=(0.6,0.6,0.6),zorder=7,label='Post samples')
ax3.plot(x_p,y_posterior_lower,color='g',linewidth=2.0,linestyle='--',zorder=9)
ax3.plot(x_p,y_posterior_samples,color=(0.8,0.8,0.8),zorder=1)
# ax3.plot(x_data,y_data,color='b',marker='.',linestyle='',label='Data')


ax3.set_yticks([])
ax3.set_xlabel(x_lab)
ax3.set_ylabel(y_lab)
ax3.legend(frameon=True,fontsize=12)

f.show()






#* Linear mapping
# lm = lambda a,b,x: (x - b)/a
lm = lambda a,b,x: a*x + b
EBM_stages = np.unique(stages_long[bl])
t_EBM_function = lm(a,b,EBM_stages) #(EBM_stages - b_)/a_
t = t_EBM_function[1:] - t_EBM_function[0]

t_p = []
for k in range(len(samples_robust['a'])):
    ay = samples_robust['a'][k]
    be = samples_robust['b'][k]
    tee = lm(ay,be,EBM_stages)
    t_p.append( tee[1:] - tee[0] )

from statsmodels import robust
t = np.nanmedian(t_p,axis=0)
t_max = t + robust.mad(t_p,axis=0)
t_min = t - robust.mad(t_p,axis=0)

#* Plot the fits
ex = x_p #np.linspace(min(x_data), max(x_data))
# f,(ax1, ax2,ax3)= plt.subplots(1,3,sharey=True,figsize=(16,7))
# for k in range(len(samples['a'])):
#     wy = samples['a'][k] * ex + samples['b'][k]
#     wy_robust = samples_robust['a'][k] * ex + samples_robust['b'][k]
#     ax1.plot(ex,wy,color=[0.8,0.8,0.8])
#     ax2.plot(ex,wy_robust,color=[0.8,0.8,0.8])
#
# ax1.plot(ex,wy,color=[0.8,0.8,0.8],label='Posterior Samples')
# ax2.plot(ex,wy_robust,color=[0.8,0.8,0.8],label='Posterior Samples')
#
# wy_mean = samples['a'].mean() * ex + samples['b'].mean()
# wy_mean_robust = samples_robust['a'].mean() * ex + samples_robust['b'].mean()
# ax1.plot(x_data, y_data, "o", label='MC Data')
# ax2.plot(x_data, y_data, "o", label='MC Data')
# ax1.plot(ex, wy_mean, "r-", label='Fit')
# ax2.plot(ex, wy_mean_robust, "r-", label='Robust Fit')
# ax1.legend(frameon=True,fontsize=12)
# ax2.legend(frameon=True,fontsize=12)
# ax1.set_xlabel(x_lab)
# ax1.set_yticks([])
# ax2.set_yticks([])
# ax2.set_xlabel(x_lab)
# ax1.set_ylabel(y_lab)

f,ax3 = plt.subplots(1,1,figsize=(10,5))
ax3.plot(x_p,y_posterior_middle,color='k',linewidth=2.0,linestyle='-',zorder=1,label='GP posterior mean. RMSE = {0}'.format(round(RMSE,2)))
ax3.plot(x_p,y_posterior_upper,color='r',linewidth=2.0,linestyle='--',zorder=2,label='+/- std')
ax3.plot(x_p,y_posterior_samples[:,1],color=(0.8,0.8,0.8),zorder=3,label='Post samples')
ax3.plot(x_p,y_posterior_lower,color='r',linewidth=2.0,linestyle='--',zorder=4)
ax3.plot(x_p,y_posterior_samples,color=(0.8,0.8,0.8),zorder=0)
ax3.plot(x_data,y_data,color='b',marker='.',linestyle='',label='Data')
ax3.set_yticks([])
ax3.set_xlabel(x_lab)
ax3.set_ylabel(y_lab)
ax3.legend(frameon=True,fontsize=12)

f.show()

f.savefig('{0}-EYOvsEBMfitGPR-20181106.png'.format(nom),dpi=300)




#*** data-driven sigmoids: sampled only at stages having data
xp = np.linspace(min(x_data), max(x_data), np.max(x_data)+1)

credible_interval = 0.50
t_EBM_samples, t_EBM_middle, t_EBM_upper, t_EBM_lower, rmse = ebm_ute.evaluate_GP_posterior(xp,x_data,y_data,rho_sq_GP,eta_sq_GP,sigma_sq_GP,CredibleIntervalLevel=credible_interval)

t_EBM = np.cumsum(np.diff(t_EBM_middle))
t_EBM_u = np.cumsum(np.diff(t_EBM_upper))
t_EBM_l = np.cumsum(np.diff(t_EBM_lower))

EBM_stages_ = np.linspace(min(EBM_stages),max(EBM_stages),num=(1+max(EBM_stages))).astype(int)
# plt.close('all')
fig,ax = plt.subplots(1,2,figsize=(12,6),sharey=False)
ax[0].boxplot(t_EBM_samples.T,notch=True,positions=xp) #EBM_stages_)
ax[0].set_ylabel('EYO')
ax[0].set_xlabel('EBM Stage')
ax[1].fill_between(xp[1:],t_EBM_l,t_EBM_u,facecolor='gray',alpha=0.3)

ax[1].plot(xp[1:],t_EBM,'.')
# ax[1].plot(EBM_stages[1:],t_EBM_u,'k:',alpha=0.5,label='50% cred int.')
# ax[1].plot(EBM_stages[1:],t_EBM_l,'k:',alpha=0.5)
# ax[1].legend(fontsize=15)
ax[1].set_title('Cumulative Time to Event',fontsize=15)
ax[1].set_ylabel('Years',fontsize=15)
ax[1].set_xlabel('EBM stage',fontsize=15)
fig.show()

######
# plt.close('all')
PeeVeeDee = pvd_[:][seq_]
PeeVeeDee = np.concatenate( (np.zeros((PeeVeeDee.shape[0],1)), PeeVeeDee) ,axis=1) # Add zeros for stage 0
#* Keep only the columns having staged data
PeeVeeDee = PeeVeeDee[:,EBM_stages]
OxCurves = np.cumsum(PeeVeeDee,axis=1).T 
OxCurves = OxCurves / np.max(OxCurves.flatten())
labs = [events_labels[k] for k in seq_] #[ebm_scores_labels[k] for k in seq_]
xCurves = EBM_stages
# ml_orders_bootstrap_fixedMM

tCurves = np.concatenate( (np.zeros((1,)),t_EBM), axis=0)
tCurves_u = np.concatenate( (np.zeros((1,)),t_EBM_u), axis=0)
tCurves_l = np.concatenate( (np.zeros((1,)),t_EBM_l), axis=0)

fig,ax = plt.subplots(1,3,figsize=(16,6),sharey=True)
ax[0].plot(xCurves,OxCurves,'d-') #,label=labs)
ax[0].set_xlabel('EBM stage',fontsize=18)
ax[0].set_ylabel('Cumulative abnormality',fontsize=18)
ax[0].legend()
ax[1].plot(tCurves,OxCurves,'.-')
ax[1].plot(tCurves_l,OxCurves,':',alpha=0.2)
ax[1].plot(tCurves_u,OxCurves,':',alpha=0.2)
ax[1].set_xlabel('Years to Event',fontsize=18)
ax[2].plot(t_EBM_middle,OxCurves,'.-')
ax[2].set_xlabel('EYO',fontsize=18)
ax[2].legend(labs,fontsize=18,loc='best')

ax[0].tick_params(axis='both', which='major', labelsize=16)
ax[1].tick_params(axis='both', which='major', labelsize=16)
ax[2].tick_params(axis='both', which='major', labelsize=16)

fig.show()


fig,ax = plt.subplots(1,1,figsize=(20,9))
ax.plot(t_EBM_middle,OxCurves,'.-')
ax.set_xlabel('EYO',fontsize='xx-large')
ax.set_ylabel('Cumulative abnormality',fontsize='xx-large')
ax.legend(labs,fontsize='large',loc=2,framealpha=1)
ax.tick_params(axis='both', which='major', labelsize='x-large')
fig.show()
# fig.savefig('{0}-sigmoids-{1}.png'.format(nom,rundate),dpi=300)

    # #*** end main()
    # return 0


#
# if __name__ == "__main__":
#     main()



#* Classification analysis
P = sum(df_staging.loc[bl,'MC']==1)
N = sum(df_staging.loc[bl,'MC']==0)
TP = []
TN = []
FP = []
FN = []
for k in range(n_biomarkers):
    TP.append(sum(df_staging.loc[bl & (df_staging.MC==1),'Model stage'][df_staging.MC[bl]==1]>=k))
    TN.append(sum(df_staging.loc[bl & (df_staging.MC==0),'Model stage'][df_staging.MC[bl]==0]<k))
    FN.append(sum(df_staging.loc[bl & (df_staging.MC==1),'Model stage'][df_staging.MC[bl]==1]<k))
    FP.append(sum(df_staging.loc[bl & (df_staging.MC==0),'Model stage'][df_staging.MC[bl]==0]>=k))
ACC = [(tp + tn)/(P + N) for (tp,tn) in zip(TP,TN)]


fig,ax = plt.subplots(figsize=(12,6))
ax.plot(ACC,'x:',label='Accuracy',markersize=12,linewidth=3)
ax.plot(TP/P,'+:',label='Sensitivity',markersize=8)
ax.plot(TN/N,'d:',label='Specificity',markersize=8)
ax.set_xlabel('Model stage')
ax.set_title('Classifying fAD patients')
ax.legend(loc='lower center')
fig.show()
fig.savefig('{0}-Classifier-Stage-{1}.png'.format(nom,rundate))

fig,ax = plt.subplots(figsize=(12,6))
ax.plot(t_EBM_middle,ACC,'x:',label='Accuracy',markersize=12,linewidth=3)
ax.plot(t_EBM_middle,TP/P,'+:',label='Sensitivity',markersize=8)
ax.plot(t_EBM_middle,TN/N,'d:',label='Specificity',markersize=8)
ax.set_xlabel('EYO')
ax.set_title('Classifying fAD patients')
ax.legend(loc='lower center')
fig.show()
fig.savefig('{0}-Classifier-EYO-{1}.png'.format(nom,rundate))


