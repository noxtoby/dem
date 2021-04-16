function [output_,DEM_object_copy] = TADPOLE_Oxtoby_DEM_SampleFromPosterior(samples_struct,DEM_object,nSamplesFromGPPosterior)
%TADPOLE_Oxtoby_DEM_SampleFromPosterior Samples from GP regression posterior
%
% Neil Oxtoby, UCL, October 2017
plotting = true;

if nargin<3
  nSamplesFromGPPosterior = 500;
end

DEM_object_copy = DEM_object;

%* Extract samples
rho_sq_samples = squeeze(samples_struct.rho_sq);
rho_sq_R = psrf(rho_sq_samples); %[R,NEFF,V,W,B] = psrf(rho_sq_samples);
inv_rho_sq_samples = squeeze(samples_struct.inv_rho_sq);
inv_rho_sq_R = psrf(inv_rho_sq_samples );
eta_sq_samples = squeeze(samples_struct.eta_sq);
eta_sq_R = psrf(eta_sq_samples );
sigma_sq_samples = squeeze(samples_struct.sigma_sq);
sigma_sq_R = psrf(sigma_sq_samples);
log_lik_samples = squeeze(samples_struct.log_lik);
waic__noxtoby = waic_noxtoby(log_lik_samples(:));
waic = mstan.waic(log_lik_samples(:));
[loo,loos,pk] = psisloo(log_lik_samples(:));

output_.rho_sq_samples = rho_sq_samples;
output_.rho_sq_R = rho_sq_R;
output_.inv_rho_sq_samples = inv_rho_sq_samples;
output_.inv_rho_sq_R = inv_rho_sq_R;
output_.eta_sq_samples = eta_sq_samples;
output_.eta_sq_R = eta_sq_R;
output_.sigma_sq_samples = sigma_sq_samples;
output_.sigma_sq_R = sigma_sq_R;
output_.log_lik_samples = log_lik_samples;
output_.waic__noxtoby = waic__noxtoby;
output_.loo = loo;
output_.loos = loos;
output_.pk = pk;

fprintf('Convergence stats for GP hyperparameter estimation:\neta_sq, rho_sq, sigma_sq \nR_hat = %.3f, %0.3f, %0.3f\n',mean(eta_sq_R),mean(rho_sq_R),mean(sigma_sq_R))
fprintf('WAIC (mstan)  : %e\n',waic.waic)
fprintf('WAIC (noxtoby): %e\n',waic__noxtoby.waic)
fprintf('PSIS-LOO CV\n - LOO  : %e\n',loo)
fprintf(' - LOOS : %e\n',loos)
fprintf(' - PK   : %e\n',pk)

%* c) GP Posterior
CredibleIntervalLevel = 0.50;
stds = sqrt(2)*erfinv(CredibleIntervalLevel);
num = ~isnan(DEM_object.x);
x_data = DEM_object.x(num);
y_data = DEM_object.dxdt(num);
%*** Define Stan model
nInterpolatedDataPoints = 100;
DEM_object_copy.x_fit = linspace(min(x_data),max(x_data),nInterpolatedDataPoints);
%* Covariance matrices from kernels: @kernel_pred, @kernel_err, @kernel_obs
rho_sq_median = nanmedian(rho_sq_samples(:));
eta_sq_median = nanmedian(eta_sq_samples(:));
sigma_sq_median = nanmedian(sigma_sq_samples(:));
%* Observations - full kernel
K = DEM_object.kernel_obs(sqrt(eta_sq_median),sqrt(rho_sq_median),sqrt(sigma_sq_median),x_data);
%* Interpolation - signal only
K_ss = DEM_object.kernel_pred(sqrt(eta_sq_median),sqrt(rho_sq_median),DEM_object_copy.x_fit,DEM_object_copy.x_fit);
%* Covariance (observations & interpolation) - signal only
K_s = DEM_object.kernel_pred(sqrt(eta_sq_median),sqrt(rho_sq_median),DEM_object_copy.x_fit,x_data);
%* GP mean and covariance
%* Covariance from fit
DEM_object_copy.dxdt_fit_mean = (K_s/K)*y_data;
dxdt_fit_Sigma = (K_ss - K_s/K*K_s');
dxdt_fit_std = sqrt(diag(dxdt_fit_Sigma));
%* Covariance from data - to calculate residuals
K_data = K;
K_s_data = DEM_object_copy.kernel_pred(sqrt(eta_sq_median),sqrt(rho_sq_median),x_data,x_data);
dxdt_fit_mean_data = (K_s_data/K_data)*y_data; % mean fit at locations of data
residuals = y_data - dxdt_fit_mean_data;
RMSE = rms(residuals);
waicVec = waic__noxtoby.waic;

%* Fit quality metrics (not entirely sure of the marginal likelihood)
evidence = exp(-0.5*(y_data'/K*y_data + log(det(K)) + length(y_data)*log(2*pi)) ); % Marginal likelihood, a.k.a., evidence
%fprintf('=== Fit quality for prior_std_inv_rho_sq = %0.4g ===\n',prior_std_inv_rho_sq)
% disp(waic__noxtoby)
% disp(waic)
% fprintf('   RMSE = %0.3f\n',RMSE)

%**** Sample from the posterior - multivariate Gaussian ****
%* Diagonalise the GP posterior covariance matrix
[Vecs,Diags] = eig(dxdt_fit_Sigma);
A = real(Vecs*(sqrt(Diags)));
%* Sample
DEM_object_copy.dxdt_fit_samples = repmat(DEM_object_copy.dxdt_fit_mean,1,nSamplesFromGPPosterior) + A*randn(length(DEM_object_copy.dxdt_fit_mean),nSamplesFromGPPosterior);
if abs(nanstd(y_data)-1)<eps
  DEM_object_copy.dxdt_fit_samples = DEM_object_copy.dxdt_fit_samples*nanstd(y_data) + nanmean(y_data);
end
%* Plot the samples and credible interval
if plotting
  ef = figure('Position',[0,20,1200,700],'color','white');
  set(plot(DEM_object_copy.x_fit,DEM_object_copy.dxdt_fit_mean,'r-'),'LineWidth',2), hold on
  hold all
  % set(plot(DEM_object_copy.x_fit,DEM_object_copy.dxdt_fit_mean + stds*dxdt_fit_std,'r-'),'LineWidth',2)
  % plot(DEM_object_copy.x_fit,DEM_object_copy.dxdt_fit_samples(:,1),'Color',0.8*[1,1,1])
  % pl = plot(DEM_object_copy.x_fit,DEM_object_copy.dxdt_fit_samples,'Color',0.8*[1,1,1]);
  % for kpp=1:length(pl)
  %   set(get(get(pl(kpp),'Annotation'),'LegendInformation'),'IconDisplayStyle','off')
  % end
  [h,patchHandles] = errorbar_shadow_gradient(DEM_object_copy.x_fit,mean(DEM_object_copy.dxdt_fit_samples,2),std(DEM_object_copy.dxdt_fit_samples,0,2));
  for kpp=(1+ceil(length(patchHandles)/2)):length(patchHandles)
    set(get(get(patchHandles(kpp),'Annotation'),'LegendInformation'),'IconDisplayStyle','off')
  end
  set(get(get(h(1),'Annotation'),'LegendInformation'),'IconDisplayStyle','off')
  % pl = plot(DEM_object_copy.x_fit,DEM_object_copy.dxdt_fit_mean,'r-','LineWidth',2); set(get(get(pl,'Annotation'),'LegendInformation'),'IconDisplayStyle','off')
  % pl = plot(DEM_object_copy.x_fit,DEM_object_copy.dxdt_fit_mean + stds*dxdt_fit_std,'r-','LineWidth',2); set(get(get(pl,'Annotation'),'LegendInformation'),'IconDisplayStyle','off')
  % pl = plot(DEM_object_copy.x_fit,DEM_object_copy.dxdt_fit_mean - stds*dxdt_fit_std,'r-','LineWidth',2); set(get(get(pl,'Annotation'),'LegendInformation'),'IconDisplayStyle','off')
  set(plot(x_data,y_data,'b.'),'LineWidth',4,'MarkerSize',16)
  title({sprintf('%s: %i posterior samples',strrep(DEM_object_copy.name,'_','-'),nSamplesFromGPPosterior),''},'FontSize',24)
  xlabel('$x$ ','FontSize',24,'Interpreter','latex')
  ylabel('$$\frac{dx}{dt}$$ ~~~~~','FontSize',24,'Interpreter','latex','Rotation',0)
  legend(['Mean',strcat({'1','2','3'},'\sigma credible interval'),'Data'])
  box off
  set(gcf,'Position',get(gcf,'Position')+rand(1,4))
  pause(0.1)
  % export_fig()
  % %print('-dpsc','-r200','')
  % savefig('')
end
end