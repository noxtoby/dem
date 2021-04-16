// Fit hyperparameters for a Gaussian process prior using a squared-exponential kernel
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
  // real min_eta_sq;
  // real min_sigma_sq;
  vector[N1] mu;
  y1_mean <- mean(y1);
  for (i in 1:N1) mu[i] <- y1_mean;
  min_inv_rho_sq <- pow((max(x1) - min(x1))/10,2);
}
parameters {
  real<lower=min_inv_rho_sq> inv_rho_sq;
  real<lower=0> eta_sq;
  real<lower=0> sigma_sq;
}
transformed parameters {
  real<lower=0> rho_sq;
  rho_sq <- inv(inv_rho_sq);
}
model {
  //*** GP covariance
  matrix[N1,N1] Sigma;
  for (i in 1:(N1-1)) {
    for (j in (i+1):N1) {
      Sigma[i,j] <- eta_sq * exp(- rho_sq * pow(x1[i] - x1[j],2)); 
      Sigma[j,i] <- Sigma[i,j];
    }
  }
  for (k in 1:N1) {
    Sigma[k,k] <- eta_sq + sigma_sq;
  }
  
  //*** Priors - GP covariance parameters
  eta_sq ~ cauchy(0,1*prior_std_eta_sq);
  inv_rho_sq ~ cauchy(0,1*prior_std_inv_rho_sq);
  sigma_sq ~ cauchy(0,1*prior_std_sigma_sq);
  
  //*** GP regression = data is MV Gaussian
  y1 ~ multi_normal(mu,Sigma);
}

generated quantities {
  real log_lik;
  //log_lik = 0;
  {
    //*** Recalculate Sigma
    matrix[N1,N1] Sigma_generated;
    for (i in 1:(N1-1)) {
      for (j in (i+1):N1) {
        Sigma_generated[i,j] <- eta_sq * exp(- rho_sq * pow(x1[i] - x1[j],2)); 
        Sigma_generated[j,i] <- Sigma_generated[i,j];
      }
    }
    for (k in 1:N1) {
      Sigma_generated[k,k] <- eta_sq + sigma_sq;
    }
    log_lik <- multi_normal_log(y1 , mu, Sigma_generated);
  }
}


