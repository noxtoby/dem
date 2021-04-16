function DEM_object_copy = TADPOLE_Oxtoby_DEM_IntegrateDEFit(DEM_object)
%TADPOLE_Oxtoby_DEM_IntegrateDEFit(DEM_object)
%  After fitting a differential equation using Stan (Hamiltonian MC, NUTS),
%  use this function to integrate posterior samples to produce
%  trajectories.
%
% Neil Oxtoby, UCL, Oct 2017

%* Preliminaries
DEM_object_copy = DEM_object;
nSamplesFromGPPosterior = size(DEM_object_copy.dxdt_fit_samples,2);

%* Extract the DE model fit
y_posterior_middle = DEM_object_copy.dxdt_fit_mean;
y_posterior_samples = DEM_object_copy.dxdt_fit_samples;

%*** Integrate the fit to get a trajectory ***
xf = DEM_object_copy.x_fit(:);
%* Mean trajectory
outputTrajectory = integrateDERoutine(xf,y_posterior_middle(:),DEM_object_copy.x(:),DEM_object_copy.dxdt(:));
DEM_object_copy.t_fit = outputTrajectory.t_fit;
DEM_object_copy.x_fit = outputTrajectory.x_fit;
DEM_object_copy.dxdt_fit_mean = outputTrajectory.other.dxdt_fit_;
DEM_object_copy.validFit = outputTrajectory.valuesToPlot_fit(2:end);

%* Trajectory for each posterior sample
outputTrajectories = cell(1,nSamplesFromGPPosterior);
DEM_object_copy.t_fit_samples = cell(1,nSamplesFromGPPosterior);
DEM_object_copy.x_fit_samples = cell(1,nSamplesFromGPPosterior);
DEM_object_copy.validFit_samples = cell(1,nSamplesFromGPPosterior);
for kSP = 1:nSamplesFromGPPosterior
  outputTrajectories{kSP} = integrateDERoutine(xf,y_posterior_samples(:,kSP),DEM_object_copy.x(:),DEM_object_copy.dxdt(:));
  DEM_object_copy.t_fit_samples{kSP} = outputTrajectories{kSP}.t_fit;
  DEM_object_copy.x_fit_samples{kSP} = outputTrajectories{kSP}.x_fit;
  DEM_object_copy.validFit_samples{kSP} = outputTrajectories{kSP}.valuesToPlot_fit(2:end);
end

end

function traj = integrateDERoutine(x_fit,dxdt_fit,x_data,dxdt_data)
%traj = integrateDERoutine(x_fit,dxdt_fit,x_data,dxdt_data)
%
% Neil Oxtoby, UCL, Oct 2017

x_fit = x_fit(:); 
dxdt_fit = dxdt_fit(:);

%* Identify integration domain
%* A single trajectory implies a monotonic DE: dxdt all the same sign
dxdt_sign = sign(nanmedian(dxdt_data)); 
if dxdt_sign<0; fun = @lt; else fun = @gt; end
%* Define the domain to include bounds (usually +/- 1*SE)
domain = fun(dxdt_fit,0);
%* Identify contiguous blocks of the fit domain
%* Edges found by difference - requires padding the arrays first
domain_edges = padarray(domain(:),1,'both');
%* Difference
domain_edges = diff(domain_edges);
%* Left edge: diff(domain) = 1
blocks_left = find(domain_edges==1);
%* Right edge: diff(domain) = -1
blocks_right = find(domain_edges==-1) - 1; % -1 shift required
% if isempty(blocks_left) || isempty(blocks_right)
%   integrateSeparately = true;
%   fprintf('integrateDEFitStan.m: one of the DE fit bounds (upper or lower) was on the wrong side of zero for the full domain, so we''re integrating over a single domain.\n')
% end

%* Identify the "disease end" of the fit as the most extreme block:
%     towards max(x) for dxdt>0
%     towards min(x) for dxdt<0
valuesToPlot_fit = false(size(domain));
if not(isempty(blocks_left)) && not(isempty(blocks_right))
  valuesToPlot_fit(blocks_left(1):blocks_right(1)) = true;
else
  traj = struct('t_fit',nan,'t_fit_',nan,'x_fit',nan,'X',nan,'Y',nan,'valuesToPlot_fit',valuesToPlot_fit,'other',struct());
  return
  %error('oxtoby:integrateDEFitStan:domain','Empty integration domain.')
end

%* "Time's arrow" for DE integration: initial and final conditions
%  find(values, 1 , ['first' / 'last'] )
if dxdt_sign<0
  ic_firstlast = 'last';  fc_firstlast = 'first';
else
  ic_firstlast = 'first'; fc_firstlast = 'last';
end

%* Array positions of initial and final condition for ODE solution
ic_fit = find(valuesToPlot_fit,1,ic_firstlast);
fc_fit = find(valuesToPlot_fit,1,fc_firstlast);
%* Extract the fit from the domain
dxdt_fit_ = dxdt_fit(ic_fit:dxdt_sign:fc_fit);

% whos x_fit
% [ic_fit,dxdt_sign,fc_fit]
%*** Calculate time from credible/confidence interval for the fit
%* Direction set by median derivative sign
x_fit_ = x_fit(ic_fit:dxdt_sign:fc_fit);
%* Calculate differential & remove appropriate data point
%  (forward Euler derivative)
dx_fit_ = diff(x_fit_,1);
%* Remove final data point
ii = 1; % if ii=2, also remove first data point
%* dx
dx_fit_ = dx_fit_(ii:end);
%* x
x_fit_ = x_fit_(ii:(end-1));
%* dx/dt
dxdt_fit_ = dxdt_fit_(ii:(end-1));
% dt = dx / (dx/dt)
dt_fit_ = dx_fit_./dxdt_fit_;
%* t = cumsum(dt)
t_fit_ = cumsum(dt_fit_,1);
%figure, plot(t_fit,x_fit_)

%*** Anchor time t=0 => shift time relative to anchor, robustly
%{
      anchor = anchor_mu_sigma(1);
      [~,k_] = min(abs(x_fit_-anchor));
      %* Shift time
      t_f = t_fit_ - t_fit_(k_);
%}
t_f = t_fit_;
%*****************************************

%*** Output
traj = struct();
traj.t_fit = t_f;
traj.t_fit_ = t_fit_;
traj.x_fit = x_fit_;
traj.X = x_data;
traj.Y = dxdt_data;
traj.valuesToPlot_fit = valuesToPlot_fit; %[ic_fit,fc_fit];

other.dx_fit_ = dx_fit_;
other.dt_fit_ = dt_fit_;
other.dxdt_fit_ = dxdt_fit_;
other.dxdt_sign = dxdt_sign;
other.fun = fun;
traj.other = other;

end