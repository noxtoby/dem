import numpy as np
import pandas as pd

#### === 0. Convenience functions ===
def findStablesAndProgressors(t,diagnosis_numerical,id):
    "findStablesAndProgressors(t,diagnosis_numerical,id): Loops through to find subjects who progress clinically (and those who don't)."
    # Unique id
    id_u = np.unique(id)
    # Progressor, Stable, Reverter, Mixed
    progressor_u = np.zeros(id_u.shape, dtype=bool)
    stable_u = np.zeros(id_u.shape, dtype=bool)
    reverter_u = np.zeros(id_u.shape, dtype=bool)
    mixed_u = np.zeros(id_u.shape, dtype=bool)
    
    progression_visit_u = np.empty(id_u.shape)
    progression_visit_u[:] = np.nan
    reversion_visit_u = np.empty(id_u.shape)
    reversion_visit_u[:] = np.nan
    
    n_visits_u = np.empty(id_u.shape)
    n_visits_u[:] = np.nan
    
    progressor = np.zeros(id.shape, dtype=bool)
    stable = np.zeros(id.shape, dtype=bool)
    reverter = np.zeros(id.shape, dtype=bool)
    mixed = np.zeros(id.shape, dtype=bool)
    
    progression_visit = np.zeros(id.shape, dtype=bool)
    reversion_visit = np.zeros(id.shape, dtype=bool)
    
    n_visits = np.empty(id.shape)
    
    # Loop through id and identify subjects who progress in diagnosis
    for k in range(0,len(id_u)):
        rowz = id==id_u[k]
        tee = t[rowz]
        dee = diagnosis_numerical[rowz]
        rowz_f = np.where(rowz)[0]
        
        #= Missing data: should be superfluous (you should've handled it already)
        not_missing = np.logical_and( np.isnan(dee)==False , np.isnan(tee)==False )
        dee = dee[not_missing]
        tee = tee[not_missing]
        rowz_f = rowz_f[not_missing]
        
        #= Number of visits
        n_visits_u[k] = len(dee) #sum( not_missing )
        
        #= Order diagnosis in time
        ordr = np.argsort(tee)
        dee = dee[ordr]
        
        # if not( (type(dee[-1])==float) | (type(dee[-1])==int)):
        #     print('Non-numeric DXNUM for RID {0}'.format(id_u[k]))
        
        #= if longitudinal data exists for this individual
        if len(dee)>1:
            dee_diff = np.diff(dee)
            if all(dee_diff>=0) & any(dee_diff>0):
                #= if Progressor
                progressor_u[k] = True
                #=== Identify progression visits ===
                pv = np.where(dee_diff>0)[0] # all visits where progression occurs
                progression_visit_u[k] = pv[0] + 1 # +1 to account for the np.diff()
                progression_visit[rowz_f[np.int(progression_visit_u[k])]] = True
            elif all(dee_diff==0):
                #= if Stable
                stable_u[k] = True
            elif all(dee_diff<0): 
                #= if Reverter
                reverter_u[k] = True
                #=== Identify reversion visits ===
                rv = np.where(dee_diff<0)[0] # all visits where reversion occurs
                reversion_visit_u[k] = rv[0] + 1 # +1 to account for the np.diff()
                reversion_visit[rowz_f[np.int(reversion_visit_u[k])]] = True
            else:
                #= if mixed diagnosis (both progression and reversion)
                mixed_u[k] = True
            
        
        #=== Propagate individual data back to original shape vectors ===
        n_visits[rowz] = n_visits_u[k]
        progressor[rowz] = progressor_u[k]
        stable[rowz] = stable_u[k]
        reverter[rowz] = reverter_u[k]
        mixed[rowz] = mixed_u[k]
        
    
    return stable, progressor, reverter, mixed, progression_visit, reversion_visit, stable_u, progressor_u, reverter_u, mixed_u, progression_visit_u, reversion_visit_u


def ismember(A,B):
    "ismember(A,B): Recursive form of np.logical_or to test if A is in B"
    # First comparison
    C = A==B[0]
    if len(B)>1:
        for k in range(1,len(B)):
            C = np.logical_or(C,A==B[k])
    return C


def integrateDERoutine(x_fit,dxdt_fit,x_data,dxdt_data,integrationDomain):
    "integrateDERoutine: integrates a DEM fit and returns a trajectory."
    # #*** Identify integration domain
    # # A single trajectory implies a monotonic DE: dxdt all the same sign
    # dxdt_sign = np.sign(np.median(dxdt_data))
    # if dxdt_sign<0:
    #     fun = lambda x: np.less(x,0)
    # else:
    #     fun = lambda x: np.greater(x,0)
    # domain = fun(dxdt_fit)
    
    dxdt_sign = np.sign(integrationDomain[-1]-integrationDomain[0])
    #*** Calculate differential (forward Euler derivative)
    x_fit_ = x_fit[integrationDomain]
    dxdt_fit_ = dxdt_fit[integrationDomain]
    dxdt_fit_ = np.delete(dxdt_fit_, [len(dxdt_fit_)-1])
    dx_fit_ = np.diff(x_fit_,1)
    x_fit_ = np.delete(x_fit_, [len(x_fit_)-1])
    
    # dt = dx / (dx/dt)
    dt_fit_ = np.divide(dx_fit_,dxdt_fit_)
    # t = cumsum(dt)
    t_fit_ = np.cumsum(dt_fit_,axis=0)
    
    return t_fit_, x_fit_

def selectBiomarker(df,markerString):
    """x_raw,t_raw = selectBiomarker(df,markerString)"""
    idString = 'RID'
    timeString = 'Years.bl'
    id_ = df[idString].as_matrix()
    t = df[timeString].as_matrix()
    x = df[markerString].as_matrix()
    # FreeSurfer volumes should be divided by intracranial volume
    icv = df['ICV.bl'].as_matrix()
    FSvolumes = ['WholeBrain','Hippocampus','Entorhinal','Ventricles','MidTemp','Fusiform']
    if markerString in FSvolumes:
        x = x/icv
        print('Biomarker x has been normalised. {0}/ICV'.format(markerString))
    
    # Remove nan
    isnan = np.isnan(x) | np.isnan(t)
    x = np.delete(x,np.where(isnan)[0])
    t = np.delete(t,np.where(isnan)[0])
    id_ = np.delete(id_,np.where(isnan)[0])
    
    rowz = ~isnan
    
    return x, t, id_, rowz

def calculateDerivatives(x,t,id):
    """
    dxdt, x0, id_, x_mean = calculateDerivatives(x,t,id)
    
    Missing data is assumed to be encoded as np.nan
    
    """
    nm = ~np.isnan(t) & ~np.isnan(x) # not missing
    id_u = np.unique(id)
    id_ = []
    dxdt = []
    x0 = []
    x_mean = []
    for k in range(0,len(id_u)):
        rowz = id==id_u[k]
        rowz = rowz & nm
        t_k = t[rowz]
        x_k = x[rowz]
        if np.sum(rowz)>1:
            # Gradient via linear regression
            lm = np.polyfit(t_k,x_k,1)
            id_.append(id_u[k])
            dxdt.append(lm[0])
            x0.append(lm[1])
            x_mean.append(np.nanmean(x_k))
            print('k = {0} \n * n = {1}\n * dx/dt = {2} | x0 = {3} | mean(x) = {4}'.format(k,sum(rowz),dxdt[-1],x0[-1],x_mean[-1]))
            #plt.plot(t[rowz],x[rowz],'x')
            #plt.plot([min(t[rowz]),max(t[rowz])],[min(t[rowz]),max(t[rowz])]*dxdt[-1] + x0[-1],'-')
            #plt.show()
    
    # Remove any nan
    dxdt_isnan = np.isnan(dxdt)
    x0_isnan = np.isnan(x0)
    dxdt = np.delete(dxdt,np.where(dxdt_isnan | x0_isnan)[0])
    x0 = np.delete(x0,np.where(dxdt_isnan | x0_isnan)[0])
    id_u = np.delete(id_u,np.where(dxdt_isnan | x0_isnan)[0])
    
    return dxdt, x0, id_, x_mean

def fitDEM(x_,dxdt_,covariates,gpdem_code='gpfit.stan'):
    """
    DEMfit = fitDEM(x_,dxdt_,covariates)
    """
    #* 0. Adjust for covariates (stepwise?)
    
    #* 1. Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_,dxdt_)
    DEMfit = {'linreg_slope':slope}
    DEMfit['linreg_intercept'] = intercept
    DEMfit['linreg_r_value'] = r_value
    DEMfit['linreg_p_value'] = p_value
    DEMfit['linreg_std_err'] = std_err
    #* 2. GP regression
    # Preliminaries
    x2 = x_*2
    y2 = dxdt_*2
    scaleFactor = 1
    inv_rho_sq_scale = (max(x2)-min(x2))/scaleFactor
    eta_sq_scale = (max(y2)-min(y2))/scaleFactor
    sigma_sq_scale = 0.1*eta_sq_scale
    # GP priors: hyperparameter scales
    cauchyHWHM_inv_rho_sq = inv_rho_sq_scale
    cauchyHWHM_eta_sq = eta_sq_scale
    cauchyHWHM_sigma_sq = sigma_sq_scale
    prior_std_inv_rho_sq = cauchyHWHM_inv_rho_sq
    prior_std_eta_sq = cauchyHWHM_eta_sq
    prior_std_sigma_sq = cauchyHWHM_sigma_sq
    #* MCMC CHAINS: initial values
    # inv_rho_sq = prior_std_inv_rho_sq # length rho ~ x
    # eta_sq = prior_std_eta_sq # signal ~ y
    # sigma_sq = prior_std_sigma_sq # var(noise) ~ 0.1*std(signal)^2
    gpdem_dat = {'N1': len(x_),
                 'x1': x_,
                 'y1': dxdt_,
                 'prior_std_eta_sq' : prior_std_eta_sq,
                 'prior_std_inv_rho_sq' : prior_std_inv_rho_sq,
                 'prior_std_sigma_sq' : prior_std_sigma_sq
                }
    fit = pystan.stan(file=gpdem_code, data=gpdem_dat,
                      iter=1000, chains=4)
    DEMfit['GPfit'] = fit
    return DEMfit
