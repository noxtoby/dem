classdef DEM < handle
  %DEM Differential Equation Model
  %  For estimating a long-term group-level trajectory from
  %  individual-level short-interval differential data, which is
  %  interpreted as gradient samples of the average trajectory.
  %  
  % Dependencies (FIXME: there may be others):
  %   nanplotTall
  %   RData
  %   MatlabStan -> CmdStan, MatlabProcessManager
  %
  % Author: Neil Oxtoby, UCL, October 2017
  % Developed in MATLAB version 8.6.0.267246 (R2015b)
  
  properties (Constant=true)
    gp = true; % 
    nChains = 4;
    nSamples = 5000;
    nWarmup = 2000;
    nThin = 2;
    defaultStanFit = StanFitOx('model',StanModel('file','gpfit.stan','chains',DEM.nChains,'iter',DEM.nSamples,'warmup',DEM.nWarmup,'thin',DEM.nThin,'verbose',true));
  end
  properties (GetAccess='private',SetAccess='private')
    % left
    % right
  end
  properties % Leave public, as get methods in other languages are public
    name
    xi       % longitudinal data
    ti       % time
    id       % individual ID
    included % subset to include in the fit
    
    x    % average over xi per individual
    x_id % id for x
    dxdt % rate of change: linear fit xi(ti)
    t_interval % time interval covered by each individual's data
    
    fitt % StanFit object and model with some defaults
    file = 'gpfit.stan';
    
    %%% GP regression kernels
    
    kernel_pred = @(eta,rho,x_1,x_2) eta^2*exp(-rho^2*(repmat(x_1(:),1,length(x_2)) - repmat(x_2(:).',length(x_1),1)).^2); % GP kernel - prediction
    kernel_err = @(sigma,x_1) sigma^2*eye(length(x_1)); % GP kernel - error
    kernel_obs = @(eta,rho,sigma,x_1) eta^2*exp(-rho^2*(repmat(x_1(:),1,length(x_1)) - repmat(x_1(:).',length(x_1),1)).^2) + sigma^2*eye(length(x_1));
    
    x_fit    % grid of x-values in DEM fit
    dxdt_fit_mean % y-values (dx/dt) in DEM fit
    dxdt_fit_samples % GP regression posterior samples
    t_fit         % time inferred from integrated DEM using GP regression average
    validFit % logical array: members of x_fit where the model is valid (montonic trajectory)
    data_R   % filename of R-dump data file for Stan
    init_R   % base filename of R-dump init files for Stan
    
    integrateGPSamples % whether to integrate GP samples, or just the mean
    x_fit_samples % x_fit grid for each posterior sample
    t_fit_samples % t_fit for each posterior sample
    validFit_samples % logical array: members of x_fit_samples where the model is valid (montonic trajectory)
    nSamplesFromGPPosterior % number of posterior samples to take from the DEM fit
  end
  
  methods 
    %% Constructor
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function self = DEM(varargin)
      p = inputParser;
      p.KeepUnmatched = true;
      p.FunctionName = 'DEM constructor';
      p.addParameter('name','',@ischar);
      p.addParameter('xi',[],@isnumeric); % longitudinal data
      p.addParameter('ti',[],@isnumeric); % time
      p.addParameter('id',[]); % individual id, per data row
      p.addParameter('included',[],@isnumeric); % subset to include in the fit
      
      %* DEM stuff
      p.addParameter('x',[],@isnumeric); % average over xi per individual
      p.addParameter('dxdt',[],@isnumeric); % rate of change: linear fit xi(ti)
      p.addParameter('x_id',[],@isnumeric); % id for x
      p.addParameter('t_interval',[],@isnumeric); % time interval covered by each individual's data
      p.addParameter('fitt',DEM.defaultStanFit,@(x)isa(x,'StanFitOx')); % StanFit object 
      p.addParameter('file','gpfit.stan',@ischar); % stan file
      
      p.addParameter('data_R','',@ischar); % filename of R-dump data file for Stan
      p.addParameter('init_R','',@ischar); % base filename of R-dump init files for Stan
      
      %* DEM fit results: not inputs
      
      p.addParameter('x_fit',[],@isnumeric); % grid of x-values in DEM fit
      p.addParameter('dxdt_fit_mean',[],@isnumeric); % y-values in DEM fit
      p.addParameter('t_fit',[],@isnumeric) % time inferred from integrated DEM-fit average
      p.addParameter('validFit','',@islogical); % members of x_fit where the model is valid (DEM fit implies monotonic trajectory)
      
      %* DEM fit posterior samples
      
      p.addParameter('integrateGPSamples','',@islogical); % whether to integrate GP samples, or just the mean
      p.addParameter('nSamplesFromGPPosterior','',@isnum); % number of samples
      p.addParameter('dxdt_fit_samples',[],@isnumeric); % y-values in DEM fit
      p.addParameter('x_fit_samples',{},@iscell) % grid of x values in DEM-fit per posterior sample
      p.addParameter('t_fit_samples',{},@iscell) % time inferred from integrated DEM-fit samples
      p.addParameter('validFit_samples',{},@iscell) % time inferred from integrated DEM-fit samples
      
      % p.addParameter('working_dir',pwd);
      % p.addParameter('file_overwrite',false,@islogical);
      
      %* Parse inputs
      p.parse(varargin{:});
      self.name = p.Results.name;
      self.xi = p.Results.xi;
      self.ti = p.Results.ti;
      self.id = p.Results.id;
      self.included  = p.Results.included;
      
      if not(isempty(p.Results.x))
        warning('Why did you pass x? Setting it to empty.')
        self.x = [];
      end
      if not(isempty(p.Results.x_id))
        warning('Why did you pass x_id? Setting it to empty.')
        self.x_id = [];
      end
      if not(isempty(p.Results.dxdt))
        warning('Why did you pass dxdt? Setting it to empty.')
        self.dxdt = [];
      end
      if not(isempty(p.Results.t_interval))
        warning('Why did you pass t_interval? Setting it to empty.')
        self.t_interval = [];
      end
      if not(isempty(p.Results.fitt))
        self.fitt = p.Results.fitt;
      else
        self.fitt = DEM.defaultStanFit;
      end
      
      if not(isempty(p.Results.data_R))
        self.data_R = p.Results.data_R;
      end
      if not(isempty(p.Results.init_R))
        self.init_R = p.Results.init_R;
      end
      if not(isempty(p.Results.file))
        self.file = p.Results.file;
      end
      
      % pass remaining inputs to set()
      %self.set(p.Unmatched);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % function set(self,varargin)
    %   p = inputParser;
    %   p.KeepUnmatched = false;
    %   p.FunctionName = 'DEM parameter setter';
    %   p.addParameter('name',self.name);
    %   p.addParameter('xi',self.xi);
    %   p.addParameter('ti',self.ti);
    %   p.addParameter('id',self.id);
    %   p.addParameter('included',self.included);
    %   p.addParameter('x',self.x);
    %   p.addParameter('dxdt',self.dxdt);
    %   p.addParameter('x_id',self.x_id);
    %   p.addParameter('t_interval',self.t_interval);
    %   p.addParameter('fitt',self.fitt);
    % 
    %   %* DEM fit results: not inputs
    %   p.addParameter('x_fit',self.x_fit);
    %   p.addParameter('dxdt_fit_mean',self.dxdt_fit_mean);
    %   p.addParameter('dxdt_fit_samples',self.dxdt_fit_samples);
    %   p.addParameter('t_fit',self.t_fit);
    %   p.addParameter('data_R',self.data_R);
    %   p.addParameter('init_R',self.init_R);
    % 
    %   p.parse(varargin{:});
    % 
    % end
    
    % function set.fitt(self,fitt)
    %   self.fitt = self.defaultStanFit;
    % end
    
    function calculateGradientTallFormat( self )
      %calculateGradientTallFormat( self )
      %   Calculate the time-gradient from
      %   x and t in tall format, given id
      %   by fitting a straight line using ordinary least squares.
      %
      %  Fits xTall(id==id(k)) = x_bl + dx_dt(k)*tTall(id=id(k))
      %  and returns:
      %    dx_dt
      %    x(k) = mean(x(id==id(k)))
      %    (optional) extras = {dt_mean,x_bl,diffx_difft}
      %                      = {average followup interval,
      %                         fitted baseline value (intercept),
      %                         finite-difference gradient: bl to first followup}
      %
      %  Author: Neil Oxtoby, UCL, Nov 2015
      %  Project: Biomarker Ecology (trajectories from cross-sectional data)
      %  Team: Progression Of Neurodegenerative Disease
      rbo = 'off';
      useRobustFittingIfSufficientData = false;
      if useRobustFittingIfSufficientData; rbo = 'on'; end
      
      %* Included subjects only
      if isempty(self.included)
        warning('Assuming all data is to be included.')
        self.included = true(size(self.xi));
      end
      
      id_u = unique(self.id);
      x_ = nan(size(id_u));
      x_bl = nan(size(x_));  % linear fit intercept
      dxdt_ = nan(size(x_)); % linear fit gradient
      sigma_x = nan(size(x_));  % linear fit residuals
      t_range = nan(size(x_));  % followup length
      for ki=1:length(id_u)
        rowz = self.id==id_u(ki);
        rowz = rowz & self.included;
        x_i = self.xi(rowz);
        t_i = self.ti(rowz);
        %* Remove missing (NaN)
        nums = ~isnan(x_i) & ~isnan(t_i);
        x_i = x_i(nums);
        t_i = t_i(nums);
        t_i = t_i - min(t_i); % shift to zero (so x_bl is intercept)
        %* Fit a straight line using OLS
        if length(x_i)>=2
          if length(x_i)>=4 % robust (if desired)
            model_poly1 = LinearModel.fit(t_i,x_i,'x ~ t','VarNames',{'t','x'},'RobustOpts',rbo);
          elseif length(x_i)<=3 % nonrobust (not enough data points)
            model_poly1 = LinearModel.fit(t_i,x_i,'x ~ t','VarNames',{'t','x'},'RobustOpts','off');
          end
          %* Mean biomarker value
          x_(ki) = mean(x_i);
          t_range(ki) = range(t_i);
          %* geometric mean of fitted values
          %x(ki) = nthroot(prod(model_poly1.Fitted),length(xi));
          %* Gradient
          dxdt_(ki) = model_poly1.Coefficients.Estimate(2);
          %* Intercept (= first fitted value - usually "baseline")
          x_bl(ki) = model_poly1.Fitted(1); %model_poly1.Coefficients.Estimate(1);
          %* Residuals standard deviation
          sigma_x(ki) = std(model_poly1.Residuals.Raw);
        end
      end
      
      %* Sort x: helps with the GP regression
      [~,I] = sort(x_,'ascend');
      self.x = x_(I);
      self.x_id = id_u(I);
      self.dxdt = dxdt_(I);
      self.t_interval = t_range(I);
    end
    
    function plotDEM( self )
      monotonicData = find(self.validFit);
      
      dxdt_sign = sign(nanmedian(self.dxdt));
      if dxdt_sign<0
        fun = @nanmax;
        ab = @lt;
        allButRemovedDataPoint = 2:length(monotonicData);
      else
        fun = @nanmin;
        ab = @gt;
        allButRemovedDataPoint = 1:(length(monotonicData)-1);
      end
      if fun(self.x)==fun(self.x_fit)
        allButRemovedDataPoint = monotonicData(allButRemovedDataPoint);
      else
        allButRemovedDataPoint = monotonicData(allButRemovedDataPoint);
      end
      % fprintf('length(monotonicData) = %f\n',length(monotonicData))
      % fprintf('size(allButRemovedDataPoint) = [%i,%i]\n',size(allButRemovedDataPoint))
      % fprintf('size(dxdt_fit_mean(allButRemovedDataPoint)) = [%i,%i]\n',size(self.dxdt_fit_mean(allButRemovedDataPoint)))
      % fprintf('size(x_fit) = [%i,%i]\n',size(self.x_fit))
      % display(allButRemovedDataPoint(:).')
      % display(monotonicData(:).')
      
      lab = strrep(self.name,'_','-');
      spl = 13; spw = 3;
      
      figure('Position',[20,20,1200,600],'Color','white')
      spv = 1:spw; % subplot stuff
      subplot(1,spl,spv)
      nanplotTall(self.ti,self.xi,self.id)
      title('Spaghetti plot'),ylabel(lab),xlabel('Study time')
      spv = spv + spw+1; % subplot stuff
      subplot(1,spl,spv)
      plot(self.dxdt,self.x,'x')
      title('DEM data'),xlabel('dx/dt')%ylabel(strcat('x = ',lab))
      if not(isempty(self.dxdt_fit_mean))
        hold all,plot(self.dxdt_fit_mean(allButRemovedDataPoint),self.x_fit(allButRemovedDataPoint),'r','LineWidth',2)
        legend('Data','Fit')
      end
      spv = 3*spw; % subplot stuff
      sp = subplot(1,spl,spv);
      axes
      ylabel('Boxplots go here')
      box off
      
      %{
      %* FIXME: add inputs to plotDEM
      %* - normals / abnormals
      %* - symptomatic / asymptomatic
      
      y = quantile(DEM_object.x,0.25*[1,2,3]);
      abnormals = ab(self.x,y(2)); normals = ~ab(self.x,y(2));
      sympto = ab(self.x,y(1)); asympto = ~ab(self.x,y(3));
      G = 4*(abnormals & sympto) + 3*(abnormals & ~sympto) + 2*(normals & sympto) + 1*(normals & ~sympto);
      G = 4*() + 3*(self.x<y(2)) + 2*(self.x<y(1)) +
      1*(self.x<y(1));
      G(G==0) = 5;
      Glab = {'aN','sN','aA','sA','other'};
      subsetToDetermineTransitionBoundaries = abnormals_symptomatic_anchor |   normals_asymptomatic_anchor;
      subsetToDetermineTransitionBoundaries = subsetToDetermineTransitionBoundaries & G~=5;
      XX = x_adj(subsetToDetermineTransitionBoundaries); %x_adj_ps(:); %vertcat(x_adj(:),x_nc_adj(:));
      GG = G(subsetToDetermineTransitionBoundaries);
      GG = GG(:); %GG(x_fv); %vertcat(G(postSelect),G(asympto_nc));
      XX = XX(:); %XX(x_fv);
      medianAbnormal = nanmedian(XX(GG==4)); %anchor_mu_sigma_sMC(1); % nanmedian(XX(GG==4));
      medianNormal = nanmedian(XX(GG==1)); %anchor_mu_sigma_aMC(1); %nanmedian(XX(GG==3));
      % EBMthreshold = ;
      gren = [0,0.7,0]; red = [1,0,0];
      boxPlotStyle = 'traditional'; % 'compact'
      h_box = boxplot(XX,GG,'colors',[gren;red],'boxstyle','outline','symbol','+','plotstyle',boxPlotStyle,'labels',Glab(unique(GG)),'widths',0.9);
      ax_box = gca;
      yl = [min(x_adj),max(x_adj)]; %get(ax_box,'ylim');
      set(gca,'ylim',yl)
      ap = get(ax_box,'Position');
      box off
      drawnow
      set(h_box,'LineWidth',2)
      % set(gca,'xticklabel',Glab(unique(GG)))
      try
        if strcmpi(boxPlotStyle,'compact')
          h = findobj(gca, 'type', 'text');
          set(h(:),'Visible','on')
          set(h(2),'Color',gren)
          set(h(1),'Color',red)
        else
          %[hx,~] = format_ticks(ax_box,'','');
          %set(hx(1),'Color',gren,'Rotation',0),set(hx(2),'Color',red,'Rotation',0)
        end
      catch
      end
      %}
      
      spv = (spl-2):spl; % subplot stuff
      subplot(1,spl,spv)
      plot(self.t_fit,self.x_fit,'k-','LineWidth',2)
      if not(isempty(self.t_fit_samples))
        ph = nan(1,length(self.t_fit_samples)); % plot handles
        for kt=1:length(self.t_fit_samples)
          hold all
          ph(kt) = plot(self.t_fit_samples{kt},self.x_fit_samples{kt},'-','LineWidth',1,'Color',0.9*[1,1,1]);
        end
      end
      title({'DEM trajectory';'(need to plot samples here, too)'})
      xlabel('DEM time')
      set(gca,'xlim',[min(self.t_fit),max(self.t_fit)])
    end
    
    function generateRDumpFiles(self)
      % Convenience function for saving r-dump data files to be used as
      % input to the compiled stan program.
      %
      % Neil Oxtoby, UCL, Nov 2017
      
      %*** Setup the data, in case text files aren't found
      %* Remove NaN
      num = not(isnan(self.x) | isnan(self.dxdt));
      %* Setup for MatlabStan
      N1 = sum(num);
      x1 = self.x(num);
      y1 = self.dxdt(num);
      %*** data/init: DEFAULTS
      %* Scale of GP hyperparameter priors (inverse Cauchy)
      scaleFactor = 1;
      inv_rho_sq_scale = range(x1.^2)/scaleFactor; % horizontal scale for GP regression: less than x-range of data
      eta_sq_scale     = range(y1.^2)/scaleFactor; % vertical scale for GP regression: less than y-range (dx/dt)
      sigma_sq_scale   = 0.1*eta_sq_scale;         % regression residual
      cauchyHWHM_inv_rho_sq = inv_rho_sq_scale;
      cauchyHWHM_eta_sq     = eta_sq_scale;
      cauchyHWHM_sigma_sq   = sigma_sq_scale;
      prior_std_inv_rho_sq = cauchyHWHM_inv_rho_sq;
      prior_std_eta_sq     = cauchyHWHM_eta_sq;
      prior_std_sigma_sq   = cauchyHWHM_sigma_sq;
      
      rdump_f = fullfile(pwd,self.data_R);
      if not(exist(rdump_f,'file'))
        fprintf('Rdump data file not found, writing: %s\n',rdump_f)
        MatlabStan_dataStruct = struct('N1',N1,'x1',x1,'y1',y1, ...
          'prior_std_inv_rho_sq',prior_std_inv_rho_sq,'prior_std_eta_sq',prior_std_eta_sq,'prior_std_sigma_sq',prior_std_sigma_sq);
        
        %*** Stan data ***
        stanData = RData(MatlabStan_dataStruct);
        stanData.type('x1') = 'vector';
        stanData.type('y1') = 'vector';
        if isstruct(stanData)
          rdump(rdump_f,stanData);
        else
          %* is RData
          stanData.rdump(rdump_f);
        end
      else
        fprintf('Rdump data file found, not writing: %s\n',self.data_R)
      end
      
      %*** Stan init: checks in-line (in-loop) ***
      %*** MCMC chains: initial values ***
      inv_rho_sq = prior_std_inv_rho_sq; % length rho ~ x
      eta_sq = prior_std_eta_sq;         % signal ~ y
      sigma_sq = prior_std_sigma_sq;     % var(noise) ~ 0.1*std(signal)^2
      stanInit = struct('sigma_sq',sigma_sq, 'eta_sq',eta_sq,'inv_rho_sq',inv_rho_sq);
      %* Multiple, diffuse ICs: logarithmically-spaced (factor of 3)
      factr_min = 3^-(self.nChains-2);
      factr_max = 3;
      factr = logspace(log10(factr_min),log10(factr_max),self.nChains); % Equivalently: 3.^( -(self.model.chains - 2) : 1 );
      init_struct(1:self.nChains) = stanInit;
      for kc=1:self.nChains; %self.fitt.model.chains
        init_struct(kc).eta_sq     = factr(kc) * stanInit.eta_sq;
        init_struct(kc).inv_rho_sq = factr(kc) * stanInit.inv_rho_sq;
        init_struct(kc).sigma_sq   = factr(kc) * stanInit.sigma_sq;
        %init_R_kc = strcat(initR,sprintf('-%i.R',kc));
        init_R_kc = strrep(self.init_R,'.R',sprintf('-%i.R',kc));
        if ~exist(init_R_kc,'file')
          fprintf('Rdump init file not found, writing: %s\n',init_R_kc)
          rdump_f = init_R_kc;
          if isstruct(init_struct(kc))
            rdump(rdump_f,init_struct(kc));
          else
            %* is RData
            init_struct(kc).rdump(rdump_f);
          end
        else
          fprintf('Rdump init file found, not writing: %s\n',init_R_kc)
        end
      end
    end
    
    function loadSamples(self)
      % Load Stan-generated samples into the DEM object's StanFit object
      %
      % Neil Oxtoby, UCL, Nov 2017
      
      if numel(self.fitt.output_file) ~= self.nChains
        error('DEM:loadSamples:nChains','It appears as though the number of chains doesn''t match the number of output files.')
      end
      
      self.fitt.exit_value = zeros(1,self.fitt.model.chains);
      
      self.fitt.load_samples;
      
      % sim = mcmc();
      % for ks=1:self.nChains
      %   [hdr,flatNames,flatSamples,~] =  mstan.read_stan_csv(...
      %     self.fitt.output_file{ks},self.fitt.model.inc_warmup);
      % 
      %   if isempty(flatSamples)
      %     disp('I couldn''t find the samples CSV file: %s',self.fitt.output_file{ks});
      %   else
      %     [names,~,samples] = mstan.parse_flat_samples(flatNames,flatSamples);
      % 
      %     % Account for thinning
      %     if self.fitt.model.inc_warmup
      %       exp_warmup = ceil(self.fitt.model.warmup/self.fitt.model.thin);
      %     else
      %       exp_warmup = 0;
      %     end
      %     exp_iter = ceil(self.fitt.model.iter/self.fitt.model.thin);
      % 
      %     % FIXME, currently remove existing chain
      %     try
      %       sim.remove(ks);
      %     catch
      %     end
      %     % Append to mcmc object
      %     sim.append(samples,names,exp_warmup,exp_iter,ks);
      %     sim.user_data{ks} = hdr;
      %   end
      % end
      % self.fitt.sim = sim; % Fails because there is no StanFit.set
    end
    
    function fitDEM_cv(self)
      %fitDEM_cv()
      %
      % To do. Not currently implemented.
      %
      % Neil Oxtoby, UCL, Oct 2017
      warning('DEM.fitDEM_cv() is not yet implemented.')
    end
    
    function fitDEM(self,verbose)
      %fitDEM
      %
      % Calls MatlabStan to fit the GP regression.
      %
      % Result:
      %   self.fitt = stan()
      %   with some GP-DEM defaults set
      %
      %  Author: Neil Oxtoby, UCL, Oct 2017
      
      if nargin<2
        verbose=true;
      end
      
      %** Generate data and init R-dump files, if necessary
      %self.generateRDumpFiles();
      %{
      %* data
      rdump_f = fullfile(pwd,self.data_R);
      if exist(rdump_f,'file')
        %fprintf('File exists: %s\n',rdump_f)
        %* Read in data to struct?
        %ds = dump2struct(rdump_f);
      else
        fprintf('File does not exist: %s\n',rdump_f)
        self.generateRDumpFiles();
      end
      %* init
      if not(isempty(self.init_R))
        %fprintf('init: %s\n',self.init_R)
        if ischar(self.init_R)
          %* Search for all init files
          [p,n,e] = fileparts(self.init_R);
          init_pattern = fullfile(p,[n '*' e]);
          d = dir(init_pattern);
          init_names = fullfile({p},{d.name});
          for ki=1:length(init_names)
            ii = dump2struct(init_names{ki});
            init_struct(ki) = ii; %self.dump2struct(self,init_names{ki});
          end
        end
      else
        initR = strrep(self.data_R,'-data.R','-init');
        [p,n,e] = fileparts(initR);
        init_pattern = fullfile(p,[n '*' e]);
        d = dir(init_pattern);
        self.init_R = strcat(initR,'.R');
      end
      
      if isempty(d)
        fprintf('Files do not exist matching: %s ...\nCreating them now.\n',init_pattern)
        self.generateRDumpFiles;
      end
      %}
      %*** END R-dump file creation ***%
      
      sampleFileMask = strrep(strrep(self.data_R,'-data.R','*Samples*.csv'),'TADPOLE','gpfit_TADPOLE');
      
      d = dir(sampleFileMask);
      self.fitt.output_file = d;
      if ~isempty(d)
        warning('Found Stan samples in CSV files. Are you sure you want to continue?')
        % FIXME: want to save a correct StanFit object
        self.loadSamples;
        return
      else
        warning('No Stan samples CSV files found. Calling MatLabStan now.')
      end
      
      %* Call MatlabStan
      % t = tic;
      if any(isspace(self.file))
        warning('DEM: call to stan() will probably fail if there are spaces in the path to the stan model file.\n')
      end
      fprintf('\n*** DEM: fitDEM() ***\n    Calling MatlabStan: stan()\n...')
      % self.fitt = stan('file',MatlabStan_model_file,'data',MatlabStan_dataStruct,'verbose',verbose...
      %   ,'chains',self.fit.model.chains,'iter',self.fit.model.iter,'warmup',self.fit.model.warmup,'thin',self.fit.model.thin...
      %   ,'init',stanInit_k,'sample_file',sampleFile);
      display(self.file)
      self.fitt = stan('file',self.file,'data',self.data_R,'verbose',verbose,'chains',self.fitt.model.chains...
        ,'iter',self.fitt.model.iter,'warmup',self.fitt.model.warmup,'thin',self.fitt.model.thin...
        ,'init',self.init_R,'sample_file',sampleFileMask);
      
      % self.fitt.model.data = self.data_R;
      % self.fitt.model.sample_file = sampleFile;
      % self.fitt.model.init = self.init_R;
      % %s = strcat(self.fitt.model.command,{' '});
      % %fprintf('%s \n',horzcat(s{:}))
      % self.fitt = stan('fit',self.fitt);
      
      % fprintf('finished!\n')
      % toc(t)
      
    end
    
    function sampleFromPosterior(self,nSamplesFromGPPosterior)
      % Samples from GP regression posterior
      %
      % Neil Oxtoby, UCL, October 2017
      testing = true;
      
      if nargin<2
        nSamplesFromGPPosterior = 500;
      end
      
      %* Extract samples - version 1
      rho_sq_samples = self.fitt.extract.rho_sq; 
      rho_sq_R = psrf(rho_sq_samples); %[R,NEFF,V,W,B] = psrf(rho_sq_samples);
      inv_rho_sq_samples = self.fitt.extract.inv_rho_sq; 
      inv_rho_sq_R = psrf(inv_rho_sq_samples );
      eta_sq_samples = self.fitt.extract.eta_sq; 
      eta_sq_R = psrf(eta_sq_samples );
      sigma_sq_samples = self.fitt.extract.sigma_sq;
      sigma_sq_R = psrf(sigma_sq_samples);
      log_lik_samples = self.fitt.extract.log_lik;
      waic__noxtoby = waic_noxtoby(log_lik_samples(:));
      waic = mstan.waic(log_lik_samples(:));
      [loo,loos,pk] = psisloo(log_lik_samples(:));
      
      %* Extract samples - version 2
      %{
      csv_filename_base = self.fitt.model.params.output.file;
      stan_samples = stan_extract_samples_from_csv(csv_filename_base); %* Could extract directly from fit1 - if runStan: fit.rho_sq, etc.
      rho_sq_samples = stan_samples.rho_sq;  rho_sq_R = stan_samples.R.rho_sq;
      inv_rho_sq_samples = stan_samples.inv_rho_sq;  inv_rho_sq_R = stan_samples.R.inv_rho_sq;
      eta_sq_samples = stan_samples.eta_sq;          eta_sq_R = stan_samples.R.eta_sq;
      sigma_sq_samples = stan_samples.sigma_sq;      sigma_sq_R = stan_samples.R.sigma_sq;
      log_lik_samples = stan_samples.log_lik;
            %* Reshape by stacking chains on top of each other
      [a,b,c] = size(rho_sq_samples);
      % rho_sq_samples = reshape(rho_sq_samples,a*c,b);
      % eta_sq_samples = reshape(eta_sq_samples,a*c,b);
      % sigma_sq_samples = reshape(sigma_sq_samples,a*c,b);
      rho_sq_samples = squeeze(rho_sq_samples);
      inv_rho_sq_samples = squeeze(inv_rho_sq_samples);
      eta_sq_samples = squeeze(eta_sq_samples);
      sigma_sq_samples = squeeze(sigma_sq_samples);
      log_lik_samples = squeeze(log_lik_samples);
      %}
      
      fprintf('Convergence stats for GP hyperparameter estimation:\neta_sq, rho_sq, sigma_sq \nR_hat = %.3f, %0.3f, %0.3f\n',eta_sq_R,rho_sq_R,sigma_sq_R)
      fprintf('WAIC (mstan)  : %e\n',waic.waic)
      fprintf('WAIC (noxtoby): %e\n',waic__noxtoby.waic)
      fprintf('PSIS-LOO CV\n - LOO  : %e\n',loo)
      fprintf(' - LOOS : %e\n',loos)
      fprintf(' - PK   : %e\n',pk)
      
      %* c) GP Posterior
      CredibleIntervalLevel = 0.50;
      stds = sqrt(2)*erfinv(CredibleIntervalLevel);
      num = ~isnan(self.x);
      x_data = self.x(num);
      y_data = self.dxdt(num);
      %*** Define Stan model
      nInterpolatedDataPoints = 100;
      self.x_fit = linspace(min(x_data),max(x_data),nInterpolatedDataPoints);
      %* Covariance matrices from kernels: @kernel_pred, @kernel_err, @kernel_obs
      rho_sq_median = nanmedian(rho_sq_samples);
      eta_sq_median = nanmedian(eta_sq_samples);
      sigma_sq_median = nanmedian(sigma_sq_samples);
      %* Observations - full kernel
      K = self.kernel_obs(sqrt(eta_sq_median),sqrt(rho_sq_median),sqrt(sigma_sq_median),x_data);
      %* Interpolation - signal only
      K_ss = self.kernel_pred(sqrt(eta_sq_median),sqrt(rho_sq_median),self.x_fit,self.x_fit);
      %* Covariance (observations & interpolation) - signal only
      K_s = self.kernel_pred(sqrt(eta_sq_median),sqrt(rho_sq_median),self.x_fit,x_data);
      %* GP mean and covariance
      %* Covariance from fit
      self.dxdt_fit_mean = (K_s/K)*y_data;
      dxdt_fit_Sigma = (K_ss - K_s/K*K_s');
      dxdt_fit_std = sqrt(diag(dxdt_fit_Sigma));
      %* Covariance from data - to calculate residuals
      K_data = K;
      K_s_data = self.kernel_pred(sqrt(eta_sq_median),sqrt(rho_sq_median),x_data,x_data);
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
      self.dxdt_fit_samples = repmat(self.dxdt_fit_mean,1,nSamplesFromGPPosterior) + A*randn(length(self.dxdt_fit_mean),nSamplesFromGPPosterior);
      if abs(nanstd(y_data)-1)<eps
        self.dxdt_fit_samples = self.dxdt_fit_samples*nanstd(y_data) + nanmean(y_data);
      end
      %* Plot the samples and credible interval
      if testing
        ef = figure('Position',[0,20,1200,700],'color','white');
        set(plot(self.x_fit,self.dxdt_fit_mean,'k-'),'LineWidth',2), hold on
        set(plot(self.x_fit,self.dxdt_fit_mean + stds*dxdt_fit_std,'r-'),'LineWidth',2)
        plot(self.x_fit,self.dxdt_fit_samples(:,1),'Color',0.8*[1,1,1])
        pl = plot(self.x_fit,self.dxdt_fit_samples,'Color',0.8*[1,1,1]);
        for kpp=1:length(pl)
          set(get(get(pl(kpp),'Annotation'),'LegendInformation'),'IconDisplayStyle','off')
        end
        pl = plot(self.x_fit,self.dxdt_fit_mean,'k-','LineWidth',2); set(get(get(pl,'Annotation'),'LegendInformation'),'IconDisplayStyle','off')
        pl = plot(self.x_fit,self.dxdt_fit_mean + stds*dxdt_fit_std,'r-','LineWidth',2); set(get(get(pl,'Annotation'),'LegendInformation'),'IconDisplayStyle','off')
        pl = plot(self.x_fit,self.dxdt_fit_mean - stds*dxdt_fit_std,'r-','LineWidth',2); set(get(get(pl,'Annotation'),'LegendInformation'),'IconDisplayStyle','off')
        set(plot(x_data,y_data,'k.'),'LineWidth',2,'MarkerSize',12)
        title({sprintf('%s: %i posterior samples',strrep(self.name,'_','-'),nSamplesFromGPPosterior),''},'FontSize',24)
        xlabel('$x$ ','FontSize',24,'Interpreter','latex')
        ylabel('$$\frac{dx}{dt}$$ ~~~~~','FontSize',24,'Interpreter','latex','Rotation',0) 
        legend('Mean',sprintf('%i%% credible interval',100*CredibleIntervalLevel),'Samples','Data')
        box off
        set(gcf,'Position',get(gcf,'Position')+rand(1,4))
        pause(0.1)
        % export_fig()
        % %print('-dpsc','-r200','')
        % savefig('')
      end
    end
    
    function predict(self,newData)
      
    end
    
    function integrateDEFitStan(self)
      %integrateDEFitStan( self)
      %   After fitting a differential equation using Stan (Hamiltonian
      %   MC), this function integrates posterior samples to produce 
      %   trajectories.
      %
      % Neil Oxtoby, UCL, Oct 2017
      
      %* Preliminaries
      default_nSamplesFromGPPosterior = 200;
      default_integrateGPSamples = false;
      default_CredibleIntervalLevel = 0.50;
      
      CredibleIntervalLevel = default_CredibleIntervalLevel;
      stds = sqrt(2)*erfinv(CredibleIntervalLevel);
      if isempty(self.nSamplesFromGPPosterior)
        self.nSamplesFromGPPosterior = default_nSamplesFromGPPosterior;
      end
      if isempty(self.integrateGPSamples)
        self.integrateGPSamples = default_integrateGPSamples;
      end
      
      %* Extract the DE model fit
      y_posterior_middle = self.dxdt_fit_mean;
      y_posterior_samples = self.dxdt_fit_samples;
      
      %*** Integrate the fit to get a trajectory ***
      xf = self.x_fit(:);
      %* Mean trajectory
      outputTrajectory = self.integrateDERoutine(xf,y_posterior_middle(:),self.x(:),self.dxdt(:));
      self.t_fit = outputTrajectory.t_fit;
      self.x_fit = outputTrajectory.x_fit;
      self.dxdt_fit_mean = outputTrajectory.other.dxdt_fit_;
      self.validFit = outputTrajectory.valuesToPlot_fit(2:end);
      
      %* Trajectory for each posterior sample
      if self.integrateGPSamples
        outputTrajectories = cell(1,self.nSamplesFromGPPosterior);
        self.t_fit_samples = cell(1,self.nSamplesFromGPPosterior);
        self.x_fit_samples = cell(1,self.nSamplesFromGPPosterior);
        self.validFit_samples = cell(1,self.nSamplesFromGPPosterior);
        for kSP = 1:self.nSamplesFromGPPosterior
          outputTrajectories{kSP} = self.integrateDERoutine(xf,y_posterior_samples(:,kSP),self.x(:),self.dxdt(:));
          self.t_fit_samples{kSP} = outputTrajectories{kSP}.t_fit;
          self.x_fit_samples{kSP} = outputTrajectories{kSP}.x_fit;
          self.validFit_samples{kSP} = outputTrajectories{kSP}.valuesToPlot_fit(2:end);
        end
      end
      
    end
    
    
    
    % function shinyDiagnosis(self)
    %   % Launch shinyStan
    %   shinystan_dot_R = {'#'' A short R script to launch ShinyStan on existing Stan samples.'
    %     '#'' '
    %     '#'' Command line usage: '
    %     '#''   r -f shiny_stan.R'
    %     '#'' '
    %     '#'' Neil Oxtoby, UCL, October 2017'
    %     ''
    %     'library("rstan")'
    %     'library("shinystan")'
    %     ''
    %     'pth = ''/Users/noxtoby/Documents/Cluster/stan/ADNI/adnimerge20170629/20171005_ClinicalProgressorsOmitted'''
    %     'mkr = ''Ventricles'''
    %     'pat = ''gpfit_Ventricles_20171009_Samples_[1-4].csv'''
    %     'dat = ''gpfit_Ventricles.data.R'''
    %     'ini = ''gpfit_Ventricles_[1-4].init.R'''
    %     ''
    %     '# Read the Stan samples'
    %     'csvfiles <- dir(path=pth,pattern = pat, full.names = TRUE)'
    %     'sf <- read_stan_csv(csvfiles)'
    %     ''
    %     '# Read in the data'
    %     'dataF <- dir(path=pth,pattern = dat, full.names = TRUE)'
    %     'data <- read_rdump(dataF)'
    %     'initF <- dir(path=pth,pattern = ini, full.names = TRUE)'
    %     'init = c(1,2,3,4)'
    %     'for (i in 1:length(initF)) {'
    %     'init[i] = read_rdump(initF[i])'
    %     '}'
    %     ''
    %     '# Create a corresponding stanfit object'
    %     'launch_shinystan(sf)'};
    %   %* Save shiny_stan.R, then call it
    %   shinystan_call = 'r -f shiny_stan.R';
    % end
    
    function anchorTrajectory(self,x_anchor)
      %anchorTrajectory(self,x_anchor)
      %
      % Shifts estimated time to a desired "anchor" point.
      %
      % Neil Oxtoby, UCL, Oct 2017
      
      % Anchor time t=0 => shift time relative to anchor, robustly
      [~,k_] = min(abs(self.x_fit-x_anchor));
      %* Shift time
      self.t_fit = self.t_fit - self.t_fit(k_);
      
      if self.integrateGPSamples
        for ks=1:self.nSamplesFromGPPosterior
          % Anchor time t=0 => shift time relative to anchor, robustly
          [~,k_] = min(abs(self.x_fit_samples{ks}-x_anchor));
          %* Shift time
          self.t_fit_samples{ks} = self.t_fit_samples{ks} - self.t_fit_samples{ks}(k_);
        end
      end
    end

  end
  
  methods (Static)
    function traj = integrateDERoutine(x_fit,dxdt_fit,x_data,dxdt_data)
      %traj = integrateDERoutine(x_fit,dxdt_fit,x_data,dxdt_data)
      %
      % Neil Oxtoby, UCL, Oct 2017
      
      x_fit = x_fit(:);
      dxdt_fit = dxdt_fit(:);
      
      %* Identify integration domain
      %* A single trajectory implies a monotonic DE: dxdt all the same sign
      dxdt_sign = sign(nanmedian(dxdt_data)); % OLD CODE: thresh = 0*dxdt_sign; % threshold for dx/dt - monotonic
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
    
    % function tShifted = anchorTrajectory(x,t,x_anchor)
    %   % Anchor time t=0 => shift time relative to anchor, robustly
    %   [~,k_] = min(abs(x-x_anchor));
    %   %* Shift time
    %   tShifted = t - t(k_);
    % end
    
    function tStage = stageIndividuals(x,id,DEM)
      %tStage = stageIndividuals(x,id,DEM)
      %
      % Aligns the fit from a DEM object to individuals' data points.
      %
      % x - data
      % id - ID for individuals
      % DEM - DEM object with existing fit
      %
      % For multiple datapoints for any individual idk, the staging uses
      % the average value of x(id==idk).
      %
      % Neil Oxtoby, UCL, Oct 2017
      
      if isempty(DEM.x_fit)
        error('DEM:stageIndividuals:unfit','DEM object has empty x_fit. Try calling DEM.fitDEM() first.')
      end
      xf = DEM.x_fit;
      tf = DEM.t_fit;
      [~,id_2,id_num] = unique(id,'stable'); % numeric ID
      id_num_u = id_num(id_2); % numeric unique ID
      tStage = nan(size(x));
      for ki=1:length(id_num_u)
        rowz = id_num_u(ki)==id_num;
        %* Match average value to closest fit value
        xi = nanmean(x(rowz));
        [~,k_] = min(abs(xi-xf));
        %* Time from the fit
        tStage(rowz) = tf(k_);
      end
    end
    
  end

end

function dataStruct = dump2struct(dump)
% Reads R-dump file and returns a struct
%
% Neil Oxtoby, UCL, October 2017
d = importdata(dump,'\n');
nom = cell(length(d),1);
value = cell(length(d),1);
dataStruct = struct();
for kd=1:length(nom)
  parts = strsplit(d{kd});
  nom{kd} = parts{1}; % variable name
  value{kd} = parts{3};% variable value, in R format
  %* Convert to numeric
  v = str2double(value{kd}); % returns NaN when data should be an array: 'variableName <- c(1,2,...,)'
  if isnan(v)
    value{kd} = strrep(strrep(value{kd},'c(','['),')',']'); % replace 'c('=>'[' and ')'=>']'
    v = str2num(value{kd});
    if length(v)==1 && isnan(v)
      fprintf('DEM.dump2struct(): str2num seems to have failed for %s',value{kd})
    else
      value{kd} = v;
    end
  else
    value{kd} = v;
  end
  dataStruct.(nom{kd}) = value{kd};
end
end
