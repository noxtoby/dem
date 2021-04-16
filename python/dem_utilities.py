import numpy as np
import pandas as pd
import os
import pystan

from sklearn.model_selection import StratifiedKFold

from matplotlib import pyplot as plt
import seaborn as sn

import statsmodels.formula.api as smf
import statsmodels.api as sm

import itertools

from datetime import datetime

def preliminaries(fname_save,d1d2='~/Code/GitHub/TADPOLE_Billabong_pyDEM/data/TADPOLE_D1_D2.csv'):
    """
    Differential Equation Model prep
    Returns a cleaned pandas DataFrame
    Author: Neil P Oxtoby, UCL, November 2018
    """
    dem_markers = ['WholeBrain', 'Hippocampus', 'Ventricles', 'Entorhinal', 'MMSE', 'ADAS11', 'FAQ']
    if os.path.isfile(fname_save):
        print('   ...Save file detected ({0}). Prep work done. Good on ya.'.format(fname_save))
        df = pd.read_csv(fname_save,low_memory=False)
        return df, dem_markers
    else:
        print('   ...Executing preliminaries() function.')
    #* Load data
    df = pd.read_csv(d1d2,low_memory=False)
    df = df.loc[~np.isnan(df.group)]
    df = df[['RID','Time','group']+dem_markers]
    df.rename(columns={'group':'DX'},inplace=True)
    df.to_csv(fname_save,index=False)
    
    return df, dem_markers

def check_for_save_file(file_name,function):
    if os.path.isfile(file_name):
        print('check_for_save_file(): File detected ({0}) - you can load data.'.format(file_name))
        #ebm_save = sio.loadmat(file_name)
        return 1
    else:
        if function is None:
            print('You should call your function')
        else:
            print('You should call your function {0}'.format(function.__name__))
        return 0

def dxdt(x,t):
    # n = np.isnan(t) | np.isnan(x)
    # lm = np.polyfit(t[~n],x[~n],1)
    #* Fit a GLM using statsmodels
    glm_formula = 'x ~ t'
    mod = smf.ols(formula=glm_formula, data={'x':x,'t':t})
    res = mod.fit()
    return res.params[1]

def dem_gradients(df,
        markers,
        fname_save,
        id_col='RID',
        t_col='Time',
        dx_col = 'DX',
        n_timepoints_min=2):
    """
    dem_gradients()
    Calculates individual gradients from longitudinal data and 
    returns a cross-section of differential data
    Neil Oxtoby, UCL, November 2018
    """
    if os.path.isfile(fname_save):
        print('   ...Save file detected ({0}). Differential data calculated. Good on ya.'.format(fname_save))
        df_dem = pd.read_csv(fname_save,low_memory=False)
        return df_dem
    else:
        print('   ...Executing dem_gradients() function.')
    
    #* Remove individuals without enough data
    counts = df.groupby([id_col]).agg(['count'])
    counts.reset_index(inplace=True)
    has_long_data = (np.all(counts>=n_timepoints_min,axis=1))
    rid_include = counts[id_col][ has_long_data ].values
    #* Add baseline DX
    counts = counts.merge(df.loc[df['Time']==0,[id_col,dx_col]].rename(columns={dx_col:dx_col+'.bl'}),on='RID')
    dxbl_include = counts[dx_col+'.bl'][ has_long_data ].values
    #* Baseline DX
    df = df.merge(df.loc[df['Time']==0,[id_col,dx_col]].rename(columns={dx_col:dx_col+'.bl'}))
    id_dxbl = df[[id_col,dx_col+'.bl']]
    #* Keep only RID included
    df_ = df.loc[ df[id_col].isin(rid_include) ]
    #* Add baseline DX
    df_ = df_.merge(id_dxbl)
    
    #* Calculate gradients
    df_dem = pd.DataFrame(data={id_col:rid_include,dx_col+'.bl':dxbl_include})
    for i in df_dem[id_col]:
        rowz = i==df_[id_col]
        rowz_dem = i==df_dem[id_col]
        t = df_.loc[rowz,t_col]
        for m in markers:
            x = df_.loc[rowz,m]
            df_dem.loc[rowz_dem,m+'-mean'] = np.mean(x)
            df_dem.loc[rowz_dem,m+'-grad'] = dxdt(x,t)
    
    df_dem.to_csv(fname_save,index=False)
    return df_dem

def dem_postselect(df_dem,markers,dx_col='DX'):
    """
    Postselects differential data as done in Villemagne 2013:
    - Omits non-progressing (negative gradient), non-abnormal (less than biomarker median of CN) differential data
    Neil Oxtoby, UCL, November 2018
    """
    dx_dict = {1:'CN',2:'MCI',3:'AD',4:'CNtoMCI',5:'MCItoAD',6:'CNtoAD',7:'MCItoCN',8:'ADtoMCI',9:'ADtoCN'}
    x_text = '-mean'
    y_text = '-grad'
    
    df_postelection = pd.DataFrame(data={'Marker':markers})
    
    #* 1. Restrict to MCI and AD - purifies, but might also remove presymptomatics in CN
    dx_included = [2,3]
    df_ = df_dem.loc[df_dem[dx_col].isin(dx_included)].copy()
    
    #* 2. Exclude normal and non-progressing
    for m in markers:
        #* 2.1 Normal threshold = median of CN (alt: use clustering)
        normal_threshold = df_dem.loc[df_dem[dx_col].isin([1]),m+x_text].median()
        #* 2.2 Non-progressing = negative gradient
        nonprogress_threshold = 0
        excluded_rows = (df_[m+x_text] < normal_threshold) & (df_[m+y_text] < nonprogress_threshold)
        
        df_postelection.loc[df_postelection['Marker']==m,'Normal-Threshold'] = normal_threshold
    
    return df_, df_postelection

def clinical_progressors(df,id_col='RID',dx_col='DX'):
    """
    NOT CURRENTLY USED
    """
    dx_dict = {1:'Stable NL',             2:'Stable MCI',                  3:'Stable: Dementia', 
               4:'Conversion: NL to MCI', 5:'Conversion: MCI to Dementia', 6:'Conversion: NL to Dementia',
               7:'Reversion: MCI to NL',  8:'Reversion: Dementia to MCI',  9:'Reversion: Dementia to NL'}
    counts2 = df.groupby([id_col,dx_col]).agg(['count'])
    counts3 = counts2.groupby([id_col]).agg('count')
    nonstable_dx = counts3[dx_col]>2
    nonreverting_dx = counts3[dx_col].isin([1,2,3,4,5,6])
    rid_progressors = counts3.loc[nonstable_dx & nonreverting_dx,id_col]
    return rid_progressors

def fit_dem(df_dem,markers,stan_model,betancourt=False):
    """
    dem_fit = fit_dem(df,markers,stan_model)
    """
    x_text = '-mean'
    y_text = '-grad'
    
    df_dem_fits = pd.DataFrame(data={'Marker':markers})
    
    # #* 1. Linear regression
    # slope, intercept, r_value, p_value, std_err = stats.linregress(x_,dxdt_)
    # DEMfit = {'linreg_slope':slope}
    # DEMfit['linreg_intercept'] = intercept
    # DEMfit['linreg_r_value'] = r_value
    # DEMfit['linreg_p_value'] = p_value
    # DEMfit['linreg_std_err'] = std_err
    
    for m in markers:
        x = df_dem[m+x_text].values
        y = df_dem[m+y_text].values
        
        i = np.argsort(x)
        x = x[i]
        y = y[i]
        
        #* GPR setup: hyperparameters, etc.
        if betancourt:
            x_scale = (max(x)-min(x))
            y_scale = (max(y)-min(y))
            sigma_scale = 0.1*y_scale
            
            x_predict = np.linspace(min(x),max(x),20)
            N_predict = len(x_predict)
            #* MCMC CHAINS: initial values
            rho_i = x_scale/2
            alpha_i = y_scale/2
            sigma_i = sigma_scale
            init = {'rho':rho_i, 'alpha':alpha_i, 'sigma':sigma_i}
            
            dem_gpr_dat = {'N': len(x),
                           'x': x,
                           'y': y,
                           'x_scale' : x_scale,
                           'y_scale' : y_scale,
                           'sigma_scale' : sigma_scale,
                           'x_predict'   : x_predict,
                           'N_predict'   : N_predict
                       }
            df_dem_fits.loc[df_dem_fits['Marker']==m,'x_predict'] = x_predict
        else:
            x2 = x**2
            y2 = y**2
            scaleFactor = 1
            inv_rho_sq_scale = (max(x)-min(x))**2/scaleFactor # (max(x**2)-min(x**2))/scaleFactor
            eta_sq_scale = (max(y)-min(y))**2/scaleFactor # (max(y**2)-min(y**2))/scaleFactor
            sigma_sq_scale = 0.1*eta_sq_scale
            # GP priors: hyperparameter scales
            cauchyHWHM_inv_rho_sq = inv_rho_sq_scale
            cauchyHWHM_eta_sq = eta_sq_scale
            cauchyHWHM_sigma_sq = sigma_sq_scale
            prior_std_inv_rho_sq = cauchyHWHM_inv_rho_sq
            prior_std_eta_sq = cauchyHWHM_eta_sq
            prior_std_sigma_sq = cauchyHWHM_sigma_sq
            #* MCMC CHAINS: initial values
            inv_rho_sq = inv_rho_sq_scale
            eta_sq = eta_sq_scale
            sigma_sq = sigma_sq_scale
            init = {'inv_rho_sq':inv_rho_sq, 'eta_sq':eta_sq, 'sigma_sq':sigma_sq}
            dem_gpr_dat = {'N1': len(x),
                           'x1': x,
                           'y1': y,
                           'prior_std_eta_sq'     : prior_std_eta_sq,
                           'prior_std_inv_rho_sq' : prior_std_inv_rho_sq,
                           'prior_std_sigma_sq'   : prior_std_sigma_sq
                       }
        
        print('Performing GPR for {0}'.format(m))
        fit = stan_model.sampling(data=dem_gpr_dat,
                                  init=[init,init,init,init],
                                  iter=1000,
                                  chains=4)
        df_dem_fits.loc[df_dem_fits['Marker']==m,'pystan_fit_gpr'] = fit
        
    return df_dem_fits



def fit_diagnostics(stan_model_fit):
    pass
    return None


def sample_from_gpr_posterior(x,y,xp,alpha,rho,sigma,
        CredibleIntervalLevel=0.95,
        nSamplesFromGPPosterior=500):
    #* GP Posterior
    stds = np.sqrt(2) * special.erfinv(CredibleIntervalLevel)
    #* Covariance matrices from kernels: @kernel_pred, @kernel_err, @kernel_obs
    def kernel_pred(alpha,rho,x_1,x_2):
        kp = alpha**2*np.exp(-rho**2 * (np.tile(x_1,(len(x_2),1)).transpose() - np.tile(x_2,(len(x_1),1)))**2)
        return kp
    def kernel_err(sigma,x_1):
        ke = sigma**2*np.eye(len(x_1))
        return ke
    def kernel_obs(alpha,rho,sigma,x_1):
        ko = kernel_pred(alpha,rho,x_1,x_1) + kernel_err(sigma,x_1)
        return ko
    #* Observations - full kernel
    K = kernel_obs(alpha=alpha,rho=rho,sigma=sigma,x_1=x)
    #* Interpolation - signal only
    K_ss = kernel_pred(alpha=alpha,rho=rho,x_1=xp,x_2=xp)
    #* Covariance (observations & interpolation) - signal only
    K_s = kernel_pred(alpha=alpha,rho=rho,x_1=xp,x_2=x)
    #* GP mean and covariance
    #* Covariance from fit
    y_post_mean = np.matmul(np.matmul(K_s,np.linalg.inv(K)),y)
    y_post_Sigma = (K_ss - np.matmul(np.matmul(K_s,np.linalg.inv(K)),K_s.transpose()))
    y_post_std = np.sqrt(np.diag(y_post_Sigma))
    #* Covariance from data - to calculate residuals
    K_data = K
    K_s_data = kernel_pred(alpha=alpha,rho=rho,x_1=x,x_2=x)
    y_post_mean_data = np.matmul(np.matmul(K_s_data,np.linalg.inv(K_data)),y)
    residuals = y1 - y_post_mean_data
    RMSE = np.sqrt(np.mean(residuals**2))

    # Numerical precision
    eps = np.finfo(float).eps

    ## 3. Sample from the posterior (multivariate Gaussian)
    #* Diagonalise the GP posterior covariance matrix
    Vals,Vecs = np.linalg.eig(y_post_Sigma)
    A = np.real(np.matmul(Vecs,np.diag(np.sqrt(Vals))))

    y_posterior_middle = y_post_mean
    y_posterior_upper = y_post_mean + stds*y_post_std
    y_posterior_lower = y_post_mean - stds*y_post_std

    #* Sample
    y_posterior_samples = np.tile(y_post_mean,reps=(nSamplesFromGPPosterior,1)).transpose() 
                        + np.matmul(A,np.random.randn(len(y_post_mean),nSamplesFromGPPosterior))
    if np.abs(np.std(y)-1) < eps:
        y_posterior_samples = y_posterior_samples*np.std(y) + np.mean(y)
    
    return (y_posterior_middle,y_posterior_upper,y_posterior_lower,y_posterior_samples)



#* Covariance matrices from kernels: @kernel_pred, @kernel_err, @kernel_obs
def kernel_pred(eta,rho,x_1,x_2):
    kp = eta**2*np.exp(-rho**2 * (np.tile(x_1,(len(x_2),1)).transpose() - np.tile(x_2,(len(x_1),1)))**2)
    return kp
def kernel_err(sigma,x_1):
    ke = sigma**2*np.eye(len(x_1))
    return ke
def kernel_obs(eta,rho,sigma,x_1):
    ko = kernel_pred(eta,rho,x_1,x_1) + kernel_err(sigma,x_1)
    return ko

from scipy import special
def evaluate_GP_posterior(x_p,x_data,y_data,rho_sq,eta_sq,sigma_sq,
                          nSamplesFromGPPosterior = 1000,
                          plotGPPosterior = True,
                          CredibleIntervalLevel = 0.95):
    #* Observations - full kernel
    K = kernel_obs(np.sqrt(eta_sq),np.sqrt(rho_sq),np.sqrt(sigma_sq),x_data)
    #* Interpolation - signal only
    K_ss = kernel_pred(np.sqrt(eta_sq),np.sqrt(rho_sq),x_p,x_p)
    #* Covariance (observations & interpolation) - signal only
    K_s = kernel_pred(np.sqrt(eta_sq),np.sqrt(rho_sq),x_p,x_data)
    #* GP mean and covariance
    #* Covariance from fit
    y_post_mean = np.matmul(np.matmul(K_s,np.linalg.inv(K)),y_data)
    y_post_Sigma = (K_ss - np.matmul(np.matmul(K_s,np.linalg.inv(K)),K_s.transpose()))
    y_post_std = np.sqrt(np.diag(y_post_Sigma))
    #* Covariance from data - to calculate residuals
    K_data = K
    K_s_data = kernel_pred(np.sqrt(eta_sq),np.sqrt(rho_sq),x_data,x_data)
    y_post_mean_data = np.matmul(np.matmul(K_s_data,np.linalg.inv(K_data)),y_data)
    residuals = y_data - y_post_mean_data
    RMSE = np.sqrt(np.mean(residuals**2))
    # Numerical precision
    eps = np.finfo(float).eps
    ## 3. Sample from the posterior (multivariate Gaussian)
    stds = np.sqrt(2) * special.erfinv(CredibleIntervalLevel)
    #* Diagonalise the GP posterior covariance matrix
    Vals,Vecs = np.linalg.eig(y_post_Sigma)
    A = np.real(np.matmul(Vecs,np.diag(np.sqrt(Vals))))

    y_posterior_middle = y_post_mean
    y_posterior_upper = y_post_mean + stds*y_post_std
    y_posterior_lower = y_post_mean - stds*y_post_std

    #* Sample
    y_posterior_samples = np.tile(y_post_mean,(nSamplesFromGPPosterior,1)).transpose() + np.matmul(A,np.random.randn(len(y_post_mean),nSamplesFromGPPosterior))
    if np.abs(np.std(y_data)-1) < eps:
        y_posterior_samples = y_posterior_samples*np.std(y_data) + np.mean(y_data)

    return y_posterior_samples, y_posterior_middle, y_posterior_upper, y_posterior_lower, RMSE




def plot_gpr_posterior(x,xp,y,y_posterior_middle,y_posterior_upper,y_posterior_lower,y_posterior_samples,lable='x'):
    fig, ax = plt.subplots(1,2)
    ax[0].subplot(121)
    ax[0].plot(xp,y_posterior_middle,color='k',linewidth=2.0,linestyle='-',zorder=1,label='GP posterior mean')
    ax[0].plot(xp,y_posterior_upper,color='r',linewidth=2.0,linestyle='--',zorder=2,label='+/- std')
    ax[0].plot(xp,y_posterior_samples[:,1],color=(0.8,0.8,0.8),zorder=3,label='Post samples')
    ax[0].plot(xp,y_posterior_lower,color='r',linewidth=2.0,linestyle='--',zorder=4)
    ax[0].plot(xp,y_posterior_samples,color=(0.8,0.8,0.8),zorder=0)
    ax[0].plot(x,y,color='b',marker='.',linestyle='',label='Data')
    ax[0].legend()
    ax[1].subplot(122)
    ax[1].plot(x,y,'b.',label="Data")
    ax[1].legend(loc=2)
    ax[1].ylabel('dx/dt')
    ax[1].xlabel(lable)
    fig.show()
    return fig, ax




#############################

def dem_staging():
    """
    Given a trained DEM, and correctly-formatted data, stage the data
    NOTE: To use CV-DEMs, you'll need to call this for each CV fold, then combine.
    Author: Neil P Oxtoby, UCL, November 2018
    """
    pass

def dem_integrate():
    pass

def dem_cv(x,
           y,
           cv_folds=StratifiedKFold(n_splits=10, shuffle=False, random_state=None)
          ):
    """
    *** WIP ***
    Run 10-fold cross-validation
        FIXME: calculate errors using the test set
    Author: Neil P Oxtoby, UCL, November 2018
    """
    pystan_fit_gpr_cv = []

    f = 0
    for train_index, test_index in cv_folds.split(x, y):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        #* Fit
        pystan_fit_k = dem_fit(x_train,y_train,events)
        #* Save
        pystan_fit_gpr_cv.append(pystan_fit_k)
        f+=1
        print('CV fold {0} of {1}'.format(f,cv_folds.n_splits))
    return pystan_fit_gpr_cv


def cv_similarity(mcmc_samples_cv,seq):
    pvd_cv = []
    for k in range(len(mcmc_samples_cv)):
        pvd, seq = extract_pvd(ml_order=seq,samples=mcmc_samples_cv[k])
        pvd_normalised = pvd/np.tile(np.sum(pvd,axis=1).reshape(-1,1),(1,pvd.shape[1]))
        pvd_cv.append(pvd_normalised)
    
    #* Hellinger distance between rows
    # => average HD between PVDs
    #   => 45 HDs across 10-folds
    hd = np.zeros(shape=(10,10))
    for f in range(len(pvd_cv)):
        for g in range(len(pvd_cv)):
            for e in range(pvd_cv[f].shape[0]):
                hd[f,g] += hellinger_distance(pvd_cv[f][e],pvd_cv[g][e])/pvd_cv[f].shape[0]
    
    cvs = 1 - np.mean(hd[np.triu_indices(hd.shape[0],k=1)]**2)
    
    return cvs



def dem_plot():
    """
    WIP
    Author: Neil P Oxtoby, UCL, November 2018
    """
    pass
    return fig, ax



def hellinger_distance(p,q):
    #hd = np.linalg.norm(np.sqrt(p)-np.sqrt(q),ord=2)/np.sqrt(2)
    #hd = (1/np.sqrt(2)) * np.sqrt( np.sum( [(np.sqrt(pi) - np.sqrt(qi))**2 for pi,qi in zip(p,q)] ) )
    hd = np.sqrt( np.sum( (np.sqrt(p) - np.sqrt(q))**2 ) / 2 )
    return hd
