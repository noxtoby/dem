#/usr/bin/env python

# from pystan import StanModel
import numpy as np
from sklearn import linear_model
# from scipy.optimize import least_squares
# # from scipy import stats #, special

class DEM:
    #DEM Differential Equation Model
    #  For estimating a long-term group-level trajectory from
    #  individual-level short-interval differential data, which is
    #  interpreted as gradient samples of the average trajectory.
    #  
    # Dependencies (FIXME: there may be others):
    #   pystan
    #
    # Author: Neil Oxtoby, UCL
    # Date:   October 2017  - Developed in MATLAB version 8.6.0.267246 (R2015b)
    #         February 2018 - Translated to Python version 3.5.2 
    
    # ============================================================
    # public methods
    # ============================================================
    # constructor
    def __init__(self):
        self.model_name = 'gpfit'
        self.model_code = '''// Gaussian Process (prior) regression: fit hyperparameters using squared exponential kernel
        data {
          int<lower=1> N1;
          vector[N1] x1;
          vector[N1] y1;
          real<lower=0> prior_std_inv_rho_sq;
          real<lower=0> prior_std_eta_sq;
          real<lower=0> prior_std_sigma_sq;
        }
        transformed data {
          real y1_mean;
          real min_inv_rho_sq;
          vector[N1] mu;
          y1_mean = mean(y1);
          for (i in 1:N1) mu[i] = y1_mean;
          min_inv_rho_sq = pow((max(x1) - min(x1))/10,2);
        }
        parameters {
          real<lower=min_inv_rho_sq> inv_rho_sq;
          real<lower=0> eta_sq;
          real<lower=0> sigma_sq;
        }
        transformed parameters {
          real<lower=0> rho_sq;
          rho_sq = inv(inv_rho_sq);
        }
        model {
          //*** GP covariance
          matrix[N1,N1] Sigma;
          matrix[N1, N1] L;
          for (i in 1:(N1-1)) {
            for (j in (i+1):N1) {
              Sigma[i,j] = eta_sq * exp(- rho_sq * pow(x1[i] - x1[j],2));
              Sigma[j,i] = Sigma[i,j];
            }
          }
          for (k in 1:N1) {
            Sigma[k,k] = eta_sq + sigma_sq;
          }
          //*** Hyperparameter Priors
          eta_sq ~ cauchy(0,1*prior_std_eta_sq);
          inv_rho_sq ~ cauchy(0,1*prior_std_inv_rho_sq);
          sigma_sq ~ cauchy(0,1*prior_std_sigma_sq);
  
          L = cholesky_decompose(Sigma);
          //*** GP regression = data is MV Gaussian
          // y1 ~ multi_normal(mu,Sigma);
          y1 ~ multi_normal_cholesky(mu,L);
        }
        generated quantities {
          real log_lik;
          {
            //*** Recalculate Sigma
            matrix[N1,N1] Sigma_generated;
            for (i in 1:(N1-1)) {
              for (j in (i+1):N1) {
                Sigma_generated[i,j] = eta_sq * exp(- rho_sq * pow(x1[i] - x1[j],2));
                Sigma_generated[j,i] = Sigma_generated[i,j];
              }
            }
            for (k in 1:N1) {
              Sigma_generated[k,k] = eta_sq + sigma_sq;
            }
            log_lik = multi_normal_lpdf(y1 | mu, Sigma_generated);
          }
        }
        '''
        self.nChains = 4
        self.nSamples = 5000
        self.nWarmup = 2000
        self.nThin = 2
    
    # to string
    def __str__(self):
        return "[" + str(self.model_name) + "," + str(self.model_code) + "]"
    
    # Accessors
    def model_code(self):
        return self.model_code
    
    def calculateGradientTallFormat(self):
        #calculateGradientTallFormat( self )
        #   Calculate the time-gradient from
        #   x and t in tall format, given id
        #   by fitting a straight line using ordinary least squares.
        #
        #  Fits xTall(id==id(k)) = x_bl + dx_dt(k)*tTall(id=id(k))
        #  and returns:
        #    dx_dt
        #    x(k) = mean(x(id==id(k)))
        #    (optional) extras = {dt_mean,x_bl,diffx_difft}
        #                      = {average followup interval,
        #                         fitted baseline value (intercept),
        #                         finite-difference gradient: bl to first followup}
        #
        #  Author: Neil Oxtoby, UCL, Nov 2015
        #  Project: Biomarker Ecology (trajectories from cross-sectional data)
        #  Team: Progression Of Neurodegenerative Disease
        rbo = 'off'
        useRobustFittingIfSufficientData = False
        if useRobustFittingIfSufficientData:
            rbo = 'on'
        id_u = np.unique(self.id)
        x_      = np.empty(id_u.shape)
        x_bl    = np.empty(x_.shape)  # linear fit intercept
        dxdt_   = np.empty(x_.shape)  # linear fit gradient
        sigma_x = np.empty(x_.shape)  # linear fit residuals
        t_range = np.empty(x_.shape)  # followup length
        for ki in range(len(id_u)):
            rowz = self.id==id_u[ki]
            x_i = self.xi[rowz]
            t_i = self.ti[rowz]
            #* Remove missing (NaN)
            nums = ~np.isnan(x_i) & ~np.isnan(t_i)
            x_i = x_i[nums]
            t_i = t_i[nums]
            t_i = t_i - np.min(t_i) # shift to zero (so x_bl is intercept)
            #* Fit a straight line using OLS
            if len(x_i)>=2:
                if (len(x_i)>=4) & (rbo=='on'):
                    # Robust linear model fit: RANSAC algorithm
                    model_poly1 = linear_model.RANSACRegressor()
                    model_poly1.fit(t_i.values.reshape(-1,1), x_i.values.reshape(-1,1))
                    inlier_mask = ransac.inlier_mask_
                    outlier_mask = np.logical_not(inlier_mask)
                else:
                    # Non-robust (not enough data points, or forced)
                    model_poly1 = linear_model.LinearRegression()
                    model_poly1.fit(t_i.values.reshape(-1,1), x_i.values.reshape(-1,1))
            
            #* Mean biomarker value
            x_[ki] = np.mean(x_i)
            t_range[ki] = np.max(t_i)-np.min(t_i)
            #* geometric mean of fitted values
            #x(ki) = nthroot(prod(model_poly1.Fitted),length(xi))
            #* Gradient
            dxdt_[ki] = model_poly1.coef_
            #* Intercept (= first fitted value - usually "baseline")
            x_bl[ki] = model_poly1.intercept_
            #* Residuals standard deviation
            residuals = x_i.values - model_poly1.predict(t_i.values.reshape(-1,1)).T
            sigma_x[ki] = np.std(residuals)
        
        self.x = x_
        self.x_id = id_u
        self.dxdt = dxdt_
        self.t_interval = t_range
    

    def fit_dem(self):
        




