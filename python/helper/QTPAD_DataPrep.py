# Simple wrapper for data preparation
#
# Input:  ADNI data as pandas dataframe
# Output: Modified dataframe
#    * handle missing data: remove individuals with missing demographics?
#    * stables and progressors (clinical diagnosis)
#
# Author: Neil Oxtoby, UCL, January 2018
# Translated and modified from earlier MATLAB code

import numpy as np
import pandas as pd
from .convenience_functions import findStablesAndProgressors
from sklearn.model_selection import StratifiedKFold

def prep_data(df, covariatesFixed = ['AGE','PTGENDER','PTEDUCAT','APOE4'], covariatesDynamic = ['COLPROT'], covariatesMRI = ['ICV','FLDSTRENG','FSVERSION']):
    #====== Covariates ======
    df_prepped = df.copy()
    covs0 = covariatesFixed + covariatesDynamic 
    #* Fix AGE
    print('\n\n****** Converting AGE to AGE + Years_bl (because AGE was actually baseline age in the QT-PAD spreadsheet prepared by Mike Donohue in June 2017) ******\n\n')
    df_prepped['AGE.bl'] = df_prepped['AGE']
    df_prepped['AGE'] = df_prepped['AGE.bl'] + df_prepped['Years.bl']
    #* Identify the rows with missing demographic/covariate data
    print('Excluding rows with missing: DX, and fixed covariates {0}\n'.format(covs0))
    missingDemographicsCovariates = df_prepped[['DX']+covs0].isnull().any(1)
    includedRows = missingDemographicsCovariates == False
    df_prepped = df_prepped[includedRows].copy()
    
    #= Numerical diagnosis
    DX = df_prepped['DX']
    DXnum = 11*(DX=='NL') + 12*(DX=='NL to MCI') + 13*(DX=='NL to Dementia') + 22*(DX=='MCI') + 23*(DX=='MCI to Dementia') + 21*(DX=='MCI to NL') + 32*(DX=='Dementia to MCI') + 33*(DX=='Dementia') 
    DXnum[DXnum==0] = np.nan
    df_prepped['DXNUM'] = DXnum
    
    #====== Clinical progression: Stable/Progressor ======
    rID = df_prepped['RID']
    years_bl = df_prepped['Years.bl']
    stable, progressor, reverter, mixed, progression_visit, reversion_visit, stable_u, progressor_u, reverter_u, mixed_u, progression_visit_u, reversion_visit_u = findStablesAndProgressors(years_bl,DXnum,rID)
    df_prepped.DX_stable = stable
    df_prepped.DX_progressor = progressor
    df_prepped.DX_progression_visit = progression_visit
    
    return df_prepped
