% TADPOLE_Oxtoby_DEM.m
%
% My submission for TADPOLE Challenge 2017: differential equation model
%
% 1. DEM_ADNI_readtable() - load data
% 2. Train models on TADPOLE data (D1-sans-D2, or LB1)
% 3. Stage in time: training and prediction (D2, LB2).
% 4. Probabilistic forecasts:
%    a) ID1 - D1 train, D2 forecast
%    b) ID5 - D1 train, D3 forecast
%    c) ID9 - D1 train, D3 custom forecast
%
% Multiple dependencies, including the following:
% * Mine (Neil Oxtoby)
%    DEM.m (my custom differential equation model class)
%    TADPOLE_Oxtoby_DEM_AnchorTrajectory.m
%    TADPOLE_Oxtoby_DEM_ForecastContinuous.m
%    TADPOLE_Oxtoby_DEM_IntegrateDEFit.m
%    TADPOLE_Oxtoby_DEM_SampleFromPosterior.m
%    TADPOLE_Oxtoby_DEM_StageIndividuals.m
%    TADPOLE_Oxtoby_readtable.m
%    waic_noxtoby, TADPOLE_Oxtoby_CoefOfVar
% * External
%    MatlabStan and/or CmdStan
%    mcmcdiag/psrf.m
% * MATLAB (only important/critical ones are listed)
%    fitlm/predict, fitrgp/predict, table class
%
% Neil Oxtoby, UCL, October/November 2017

leaderBoard = false;
removeBiomarkers = true; % currently 
removedBiomarkers = {'AV45'};

%* Apply only if leaderBoard is false
forecastOptions = {'D2','D3','D3custom'};
forecastOptionsID = {'ID1','ID5','ID9'};
fc = 3;
forecastData = forecastOptions{fc}; 
forecastID = forecastOptionsID{fc}; 

plotting = false;

%% 0. Prelims
topFolder = '/Users/noxtoby/Documents/Research/UCLPOND/Projects/201711-My-TADPOLE-Submission';
cd(fullfile(topFolder,'code'))
runDate = datestr(now,'yyyymmdd');
daysInAYear = 365.25;
try mfile = mfilename; catch err_mfilename; fprintf('Error using mfilename: %s \n',err_mfilename.message); end
if isempty(mfile); mfile = 'TADPOLE_Oxtoby_DEM'; end
if leaderBoard
  txt = '-LB';
  forecastData = 'LB2';
else
  txt = '';
end
resultsSaveFile = fullfile(topFolder,'results',[mfile,'_results',txt,'.mat']);
dataSaveFile = fullfile(topFolder,'data',[mfile,'_data',txt,'.mat']);

%% 1. Load raw data - TADPOLE_Oxtoby_readtable()
%* D1/D2
dataFile = fullfile(topFolder,'data','TADPOLE_D1_D2.csv');
dataTable_D1D2 = TADPOLE_Oxtoby_readtable(dataFile);
  dataTable_D1D2.Properties.Description = 'TADPOLE D1/D2 table created for TADPOLE Challenge 2017.';
  dataTable_D1 = dataTable_D1D2(dataTable_D1D2.D1==1,:);
  dataTable_D2 = dataTable_D1D2(dataTable_D1D2.D2==1,:);
dataTable_D3 = TADPOLE_Oxtoby_readtable(fullfile(topFolder,'data','TADPOLE_D3.csv'));
dataTable_D3_completeVisit = TADPOLE_Oxtoby_readtable(fullfile(topFolder,'data','TADPOLE_D3_completeVisit.csv'));

%****** Adjust AGE at baseline to VISITAGE ******%
dataTable_D1D2.VISITAGE = dataTable_D1D2.AGE + dataTable_D1D2.Years_bl;
%* Left Outer Join D3 to D1D2 on {RID,VISCODE}, keeping only VISITAGE from D1D2
dataTable_D3 = outerjoin(dataTable_D3,dataTable_D1D2,'Keys',{'RID','VISCODE'},'RightVariables','VISITAGE','Type','Left');
dataTable_D3_completeVisit = outerjoin(dataTable_D3_completeVisit,dataTable_D1D2,'Keys',{'RID','VISCODE'},'RightVariables','VISITAGE','Type','Left');
dataTable_D3.AGE = dataTable_D3.VISITAGE;
dataTable_D3_completeVisit.AGE = dataTable_D3_completeVisit.VISITAGE;

%* Leaderboard: LB1/LB2
dataFile = fullfile(topFolder,'data','TADPOLE_LB1_LB2.csv');
dataTable_LB1LB2 = TADPOLE_Oxtoby_readtable(dataFile);
  dataTable_LB1LB2.Properties.Description = 'TADPOLE LB1/LB2 table created for TADPOLE Challenge 2017.';
  dataTable_LB1LB2 = join(dataTable_LB1LB2,dataTable_D1D2,'Keys',{'RID','VISCODE'},'KeepOneCopy',dataTable_LB1LB2.Properties.VariableNames);
  %****** Adjust AGE at baseline to VISITAGE ******%
  dataTable_LB1LB2.AGE = dataTable_LB1LB2.VISITAGE;
  dataTable_LB1 = dataTable_LB1LB2(dataTable_LB1LB2.LB1==1,:);
  dataTable_LB2 = dataTable_LB1LB2(dataTable_LB1LB2.LB2==1,:);

%% 2. Data prep - TADPOLE_Oxtoby_DataPrep()
% Among other things, corrects baseline AGE to be visit age
[dataTable_D1_prepped,dataTable_D1_crossValidation] = TADPOLE_Oxtoby_DataPrep(dataTable_D1);
[dataTable_D2_prepped,dataTable_D2_crossValidation] = TADPOLE_Oxtoby_DataPrep(dataTable_D2);
dataTable_D3_prepped = TADPOLE_Oxtoby_DataPrepD3(dataTable_D3);
dataTable_D3_completeVisit_prepped = TADPOLE_Oxtoby_DataPrepD3(dataTable_D3_completeVisit);
[dataTable_LB1_prepped,~] = TADPOLE_Oxtoby_DataPrep(dataTable_LB1);
[dataTable_LB2_prepped,~] = TADPOLE_Oxtoby_DataPrep(dataTable_LB2);
  %* Divide MRI markers by ICV
  biomarkersMRI = {'Ventricles','Hippocampus','WholeBrain','Entorhinal','Fusiform','MidTemp'};
  biomarkersMRI_ICV = strcat(biomarkersMRI(:),'_ICV').';
  for kb=1:length(biomarkersMRI)
    m = biomarkersMRI{kb};
    dataTable_D1_prepped.(biomarkersMRI_ICV{kb}) = dataTable_D1_prepped.(m) ./ dataTable_D1_prepped.ICV;
    dataTable_D2_prepped.(biomarkersMRI_ICV{kb}) = dataTable_D2_prepped.(m) ./ dataTable_D2_prepped.ICV;
    dataTable_D3_prepped.(biomarkersMRI_ICV{kb}) = dataTable_D3_prepped.(m) ./ dataTable_D3_prepped.ICV;
    dataTable_LB1_prepped.(biomarkersMRI_ICV{kb}) = dataTable_LB1_prepped.(m) ./ dataTable_LB1_prepped.ICV;
    dataTable_LB2_prepped.(biomarkersMRI_ICV{kb}) = dataTable_LB2_prepped.(m) ./ dataTable_LB2_prepped.ICV;
  end
%* Select biomarkers of interest
biomarkers = [biomarkersMRI_ICV, ...
  {'FDG','AV45','ABETA_UPENNBIOMK9_04_19_17','PTAU_UPENNBIOMK9_04_19_17','TAU_UPENNBIOMK9_04_19_17'...
  ,'ADAS13','MMSE','MOCA','RAVLT_immediate'}];
biomarkersLabels = [strrep(biomarkersMRI_ICV,'_','/'), ...
  {'FDG','AV45','ABETA','P-TAU','T-TAU'...
  ,'ADAS13','MMSE','MOCA','RAVLT-immediate'}];

if removeBiomarkers
  removedBiomarkers = ismember(biomarkers,removedBiomarkers);
  biomarkers(removedBiomarkers) = [];
  biomarkersLabels(removedBiomarkers) = [];
end

%* Adjust for healthy covariates **** NOT USED BELOW: I use x_raw instead of x_adj
covariates = {'AGE','PTGENDER','PTEDUCAT','APOE4','COLPROT'};
covariatesMRI = {'ICV','FLDSTRENG','FSVERSION'};
  %* D1/D2
  controls_bl = dataTable_D1_prepped.normals & strcmpi(dataTable_D1_prepped.VISCODE,'bl');
    referenceGroup = controls_bl & ~dataTable_D1_prepped.MissingDemographicsCovariates;
    [dataTable_D1_adj,dataTable_D1_adjustments_LM] = TADPOLE_Oxtoby_adjustForCovariates(dataTable_D1_prepped,referenceGroup,covariates,covariatesMRI,biomarkers,biomarkersMRI_ICV);
  %* LB1
  controls_bl_LB = dataTable_LB1_prepped.normals & strcmpi(dataTable_LB1_prepped.VISCODE,'bl');
    referenceGroup_LB = controls_bl_LB & ~dataTable_LB1_prepped.MissingDemographicsCovariates;
  [dataTable_LB1_adj,dataTable_LB1_adjustments_LM] = TADPOLE_Oxtoby_adjustForCovariates(dataTable_LB1_prepped,referenceGroup_LB,covariates,covariatesMRI,biomarkers,biomarkersMRI_ICV);
  
%* Save new spreadsheets
TADPOLE_Oxtoby_DEM_csv = fullfile('../data',[mfile,'_CovariateAdjustedData.csv']);
colz = vertcat({'RID';'PTID';'VISCODE';'EXAMDATE';'D1';'D2';'Years_bl'},covariates(:),covariatesMRI(:),biomarkers(:),strcat(biomarkers(:),'_adj'),biomarkersMRI(:) ...
  ,{'MissingDemographicsCovariates';'DXNUM';'DXNUM_bl';'DXCHANGE';'VISNUM';'PTGENDERNUM';'stables';'progressors';'progressionVisit';'normals';'abnormals'});
dataTable_D1_save = dataTable_D1_adj(:,colz);
writetable(dataTable_D1_save,TADPOLE_Oxtoby_DEM_csv)
save(dataSaveFile,'dataTable_D1_save','dataTable_D1_adjustments_LM')

TADPOLE_Oxtoby_DEM_csv = fullfile('../data',[mfile,'_CovariateAdjustedData-LB.csv']);
colz{strcmpi(colz,'D1')} = 'LB1'; colz{strcmpi(colz,'D2')} = 'LB2';
dataTable_LB1_save = dataTable_LB1_adj(:,colz);
writetable(dataTable_LB1_save,TADPOLE_Oxtoby_DEM_csv)
save(dataSaveFile,'-append','dataTable_LB1_save','dataTable_LB1_adjustments_LM')

clear dataTable_LB1_save dataTable_D1_save

%% 3. Loop through biomarkers and fit DEM
omitMCIProgressors = false;
resultsRelativePath = '../results';
%* a) Check for previous results
if exist(resultsSaveFile,'file')
  [path name ext] = fileparts(resultsSaveFile);
  fprintf('Found results file: %s%s\n      in directory: %s\n=> Loading...\n',name,ext,path)
  runDEM = false;
  load(resultsSaveFile)
else
  runDEM = true;
end

if leaderBoard
  table_TrainingData = dataTable_LB1_adj;
  table_ForecastData = dataTable_LB2_prepped;
else
  table_TrainingData = dataTable_D1_adj;
  if strcmpi(forecastData,'D2')
    table_ForecastData = dataTable_D2_prepped;
  elseif strcmpi(forecastData,'D3')
    table_ForecastData = dataTable_D3_prepped;
  else
    table_ForecastData = dataTable_D3_completeVisit_prepped;
  end
end

%* b) Build the models
%* Convert this all to object-oriented version
if runDEM
  t = table_TrainingData.Years_bl;
  id = table_TrainingData.RID;
  [~,~,visit] = unique(table_TrainingData.VISNUM,'stable');
  
  %* Disease progression only (APOE4+ clinical progressors)
  clinicalProgressors = table_TrainingData.progressors; % & ;
  apoe4 = double(table_TrainingData.APOE4); apoe4 = apoe4 - min(apoe4);
  DEMsubset = (table_TrainingData.abnormals==1 | clinicalProgressors==1) & apoe4>0;
  if omitMCIProgressors
    MCIprogressors = clinicalProgressors & table_TrainingData.DXNUM_bl==22;
    DEMsubset = DEMsubset & not(MCIprogressors);
  end
  
  firstVisit = strcmpi(table_TrainingData.VISCODE,'bl');
  normals_reference = table_TrainingData.normals & firstVisit;
  symptomatic = table_TrainingData.CDRSB>=0.5;
  symptomatic_anchor = firstVisit & table_TrainingData.CDRSB>=0.5 & table_TrainingData.CDRSB<=4.0; % CDRGLOB=0.5
  %sympto_anchor = dataTable_adj.CDRSB>=4.5 & dataTable_adj.CDRSB<=9.0; % CDRGLOB=1
  %{
    From http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3409562/
    Optimal ranges of CDR-SOB scores corresponding to the global CDR scores
    were  0.5 to  4.0 for a global score of 0.5,
          4.5 to  9.0 for a global score of 1.O,
          9.5 to 15.5 for a global score of 2.0, and
         16.0 to 18.0 for a global score of 3.0.
    When applied to the validation sample, ? scores ranged from
    0.86 to 0.94 (P <.001 for all), with 93.0% of the participants
    falling within the new staging categories.
  %}
  %* Controls = CN non-progressors, asymptomatic (based on CDRSB: see above)
  normals = table_TrainingData.normals==1;
  normals = normals & ~symptomatic;
  biomarkersDEM = cell(size(biomarkers));
  %% Per biomarker: setup DEM object, calculate gradient, save R-dump files
  for km = 1:length(biomarkers)
    marker = biomarkers{km};
    markerLabel = biomarkersLabels{km};
    marker_adjusted = strcat(marker,'_adj');
    markerLabel_adjusted = strcat(markerLabel,'-adj');
    
    m = marker; l = markerLabel;
    %m = marker_adjusted; l = markerLabel_adjusted;
    
    DEM_object_matFile = fullfile('../results',[mfile txt '-' m '-DEM.mat']);
    if exist(DEM_object_matFile,'file')==2
      continue
    end
    
    %* Biomarker data
    x_raw = table_TrainingData.(marker);
    x_adj = table_TrainingData.(marker_adjusted);
    x = table_TrainingData.(m);
    [id_u,ind] = unique(id,'stable');
    
    %* Postselect data for disease specificity etc.
    %* Optional: remove high-variance (coefficient of variation > 4)
    removeLargeCoeffOfVariance = true;
    %* Coefficient of variation < 0.25 (see Bateman 2012, and possibly also Fagan 2014: longitudinal CSF)
    if removeLargeCoeffOfVariance
      x_CoV = TADPOLE_Oxtoby_CoefOfVar(x,t,id);
      coeffOfVariationPostselect = x_CoV < 0.25;
      DEM_subset = DEM_subset & coeffOfVariationPostselect;
    end
    
    %*****
    DEM_object = DEM;
    DEM_object.name = m;
    DEM_object.xi = x;
    DEM_object.ti = t;
    DEM_object.id = id;
    DEM_object.included = DEMsubset;
    DEM_object.data_R = fullfile(resultsRelativePath,strcat(mfile,txt,'-',m,'-data.R'));
    DEM_object.init_R = strrep(DEM_object.data_R,'-data.R','-init.R');
    DEM_object.fitt.model.file = 'gpfit.stan';
    
    %* Calculate individual rates of change
    DEM_object.calculateGradientTallFormat();
    
    %* Optional: remove normal & nonProgressing differential data
    removeNormalNonProgressors = true;
    if removeNormalNonProgressors
      %* Determine marker decline or not
      if nanmedian(DEM_object.dxdt)>0
        fun_progressors = @gt;
        fun_normals = @lt;
      else
        fun_progressors = @lt;
        fun_normals = @gt;
      end
      marker_progressors = fun_progressors(DEM_object.dxdt,0);
      marker_normals = fun_normals(DEM_object.x,nanmedian(DEM_object.xi(normals_reference)) - sign(nanmedian(DEM_object.dxdt))*mad(DEM_object.xi(normals_reference)));
      marker_normalNonProgressors = marker_normals & ~marker_progressors;
      fprintf('%s - Removing %i DEMsubset with normal/non-progressing biomarker values.\n\tc.f. Villemagne et al., Lancet Neurology 12, 357 (2013).\n',marker,sum(marker_normalNonProgressors))
      DEM_object.x = DEM_object.x(~marker_normalNonProgressors);
      DEM_object.dxdt = DEM_object.dxdt(~marker_normalNonProgressors);
      DEM_object.x_id = DEM_object.x_id(~marker_normalNonProgressors);
      DEM_object.t_interval = DEM_object.t_interval(~marker_normalNonProgressors);
      DEM_object.included = DEM_object.included & ~marker_normalNonProgressors;
    else
    end
    %}
    
    %* Convert to standardized "c-score" (standardized z-score using Controls, baseline visit only)
    
    % %* First-passage time thresholds: normal and abnormal
    % x_normals_anchor = x_adj(normals_reference);
    % x_abnormals_anchor = x_adj(DEMsubset & symptomatic_anchor);
    % marker_normalLevel_median = nanmedian(x_normals_anchor);
    % marker_normalLevel_mad = mad(x_normals_anchor);
    % marker_abnormalLevel_median = nanmedian(x_abnormals_anchor);
    % marker_abnormalLevel_mad = mad(x_abnormals_anchor);
    
    DEM_object.generateRDumpFiles;
    %************ Fit GP-DEM ************%
    %{
    thyme = tic;
    DEM_object.fitDEM;
    while isempty(DEM_object.fitt.exit_value) || ~all(DEM_object.fitt.exit_value==0)
      pause(42)
      fprintf('...waiting for Stan to finish. '),toc(thyme)
    end
    fprintf('...Stan finished. '),toc(thyme)
    %}
    
    %*** Save in cell array for full staging later
    biomarkersDEM{km} = DEM_object;
    save(DEM_object_matFile,'DEM_object');
  end
  %% Now submit the data files to the CS cluster and fit the models, 
  %  producing *samples.csv files, which need to be loaded into the DEM
  %  objects before proceeding.
  
  biomarkers_staging_single = cell(length(biomarkers),1);
  %% Load results and stage, etc.
  for km = 1:length(biomarkers)
    marker = biomarkers{km};
    markerLabel = biomarkersLabels{km};
    m = marker; l = markerLabel;
    DEM_object_matFile = fullfile('../results',[mfile txt '-' m '-DEM.mat']);
    
    load(DEM_object_matFile); 
    
    %* Read in results
    %{
    %* results1
    stanDate = '20171101';
    if ismember(marker,biomarkersMRI_ICV)
      stanDate = '20171102';
    end
    %}
    
    if leaderBoard
      stanDate = '20171114';
    else
      stanDate = '20171114';
    end
    
    samplesFileMask = strrep(strrep(DEM_object.data_R,'-data.R',['_' stanDate '_Samples_']),'TADPOLE','gpfit_TADPOLE');
    DEM_object.fitt.output_file = cell(1,DEM_object.nChains);
    for ks=1:DEM_object.nChains
      DEM_object.fitt.output_file{ks} = strcat(samplesFileMask,num2str(ks),'.csv');
    end
    %* MCMC Samples
    samplesFileMask_ = strcat(samplesFileMask,'*.csv');
    samples_ = stan_extract_samples_from_csv(samplesFileMask_);
    %* Posterior samples
    nSamplesFromGPPosterior = 500;
    [output_,DEM_object] = TADPOLE_Oxtoby_DEM_SampleFromPosterior(samples_,DEM_object,nSamplesFromGPPosterior);
    %* DEM trajectories
    DEM_object = TADPOLE_Oxtoby_DEM_IntegrateDEFit(DEM_object);
    %************ Anchor trajectory ************%
    anchor_visit = strcmpi(table_TrainingData.VISCODE,'bl') & DEMsubset & table_TrainingData.DXNUM_bl==33; % baseline AD, ApoE4+
    x_anchor = nanmedian(DEM_object.xi(anchor_visit));
    DEM_object = TADPOLE_Oxtoby_DEM_AnchorTrajectory(DEM_object,x_anchor);
    %*** Stage individuals **
      %* Training dataset
      s = strcmpi(table_TrainingData.VISCODE,'bl'); s = true(size(s));
      xStage = DEM_object.xi(s);
      idStage = DEM_object.id(s);
      [tStage,tStageStd,tStage_quartiles,xStage_mean]  = TADPOLE_Oxtoby_DEM_StageIndividuals(xStage,idStage,DEM_object);
      tStage_i = tStage + nanmeanCentreID(DEM_object.ti(s),idStage);
      idStage_u = unique(idStage,'stable');
      %* Forecast dataset
      s = strcmpi(table_ForecastData.VISCODE,'bl'); s = true(size(s));
      if ismember(marker,table_ForecastData.Properties.VariableNames)
        xStageForecastData = table_ForecastData.(marker); xStageForecastData = xStageForecastData(s);
      else
        xStageForecastData = nan(size(table_ForecastData,1),1);
      end
      idStageForecastData = table_ForecastData.RID(s);
      [tStageForecastData,tStageForecastDataStd,tStageForecastData_quartiles,xStageForecastData_mean]  = TADPOLE_Oxtoby_DEM_StageIndividuals(xStageForecastData,idStageForecastData,DEM_object);
      tStageForecastData_i = tStageForecastData + nanmeanCentreID(table_ForecastData.AGE(s),idStageForecastData);
      idStageForecastData_u = unique(idStageForecastData,'stable');
      %* Staging uses average biomarker value, so I'm going to find the 
      %  average date. Not perfect, but the implicit linear approximation
      %  should hold for most individuals' data (short-interval ~= linear)
      dateStageForecastData_mean = cell(size(idStageForecastData_u));
      for kd=1:length(dateStageForecastData_mean)
        rowz = (table_ForecastData.RID==idStageForecastData_u(kd)); % & ~isnan(xStageForecastData);
        if sum(rowz)>0
          dateStageForecastData_mean{kd} = datestr(mean(datenum(table_ForecastData.EXAMDATE( rowz ),'yyyy-mm-dd')),'yyyy-mm-dd');
        else
          dateStageForecastData_mean{kd} = '';
        end
      end
      %* Staging the training data
      dateStageTrainingData_mean = cell(size(idStage_u));
      for kd=1:length(dateStageTrainingData_mean)
        rowz = (table_TrainingData.RID==idStage_u(kd)); % & ~isnan(xStage);
        if sum(rowz)>0
          dateStageTrainingData_mean{kd} = datestr(mean(datenum(table_TrainingData.EXAMDATE( rowz ),'yyyy-mm-dd')),'yyyy-mm-dd');
        else
          dateStageTrainingData_mean{kd} = '';
        end
      end
      %* Store individual staging results
      biomarkers_staging_single{km} = struct('DEM_object',DEM_object ...
        ,'xStage',xStage,'idStage',idStage,'tStage',tStage,'tStageStd',tStageStd ...
        ,'tStage_quartiles',tStage_quartiles,'xStage_mean',xStage_mean ...
        ,'xStageForecastData',xStageForecastData,'idStageForecastData',idStageForecastData,'tStageForecastData',tStageForecastData,'tStageForecastDataStd',tStageForecastDataStd ...
        ,'tStageForecastData_quartiles',tStageForecastData_quartiles,'xStageForecastData_mean',xStageForecastData_mean ...
        ,'dateStageForecastData_mean',{dateStageForecastData_mean},'dateStageTrainingData_mean',{dateStageTrainingData_mean});
      
    %* Example plot of staging results, by baseline diagnosis
    ord = {'CN','SMC','EMCI','LMCI','AD'};
    if plotting
      figure
      G = categorical(table_TrainingData.DX_bl,ord); 
      subplot(3,2,[1,3])
      p = plot(DEM_object.t_fit,DEM_object.x_fit,'r','LineWidth',4);
      hold all
      nanplotTall(tStage_i,xStage,idStage)
      legend('DEM Fit','Staged Training Data')
      title(markerLabel)
      uistack(p,'top')
      subplot(3,2,5)
      boxplot(tStage,G,'Orientation','horizontal')
      grid on
      xlabel('Disease time')
      xl = get(gca,'xlim');subplot(3,2,[1,3]),set(gca,'xlim',xl)

      G = categorical(table_ForecastData.DX_bl,ord);
      subplot(3,2,[2,4])
      p = plot(DEM_object.t_fit,DEM_object.x_fit,'r','LineWidth',4);
      hold all
      nanplotTall(tStageForecastData_i,xStageForecastData,idStageForecastData)
      legend('DEM Fit','Staged Forecast data')
      title(markerLabel)
      uistack(p,'top')
      subplot(3,2,6)
      boxplot(tStageForecastData,G,'Orientation','horizontal')
      grid on
      xlabel('Disease time')
      xl = get(gca,'xlim');subplot(3,2,[2,4]),set(gca,'xlim',xl)
    %***********************
    end
    
  end
  
end


%% 
%* 4. Generate forecasts
dateFmt = 'yyyy-mm-dd';
if leaderBoard
  v = 'v3';
  startDatenum = datenum('2010-05-01');
  nForecasts = 7*12; % per individual in Forecast dataset
  TADPOLE_Submission_csv = ['TADPOLE_Submission_Leaderboard_Billabong_DEM0',v,'.csv'];
  TADPOLE_Submission_csv_simple = ['TADPOLE_Submission_Leaderboard_Billabong_DEM1',v,'.csv'];
else
  if any(removedBiomarkers)
    v = 'b';
  else
    v = 'a';
  end
  startDatenum = datenum('2018-01-01');
  nForecasts = 5*12; % per individual in Forecast dataset
  TADPOLE_Submission_csv = ['TADPOLE_Submission_Billabong_DEM0',v,'_',forecastID,'.csv'];
  TADPOLE_Submission_csv_simple = ['TADPOLE_Submission_Billabong_DEM1',v,'_',forecastID,'.csv'];
end

RIDForecastingData_u = unique(table_ForecastData.RID);
nIndividuals_ForecastData = length(RIDForecastingData_u);
tForecast_since_startDate = ( 0:(nForecasts-1) )/12; % approximate number of years since start date

nIndividuals_TrainingData = length(idStage_u);

%*** a) combine staging across biomarkers
%* Prediction set
tStageForecastData_quartiles_all = nan(nIndividuals_ForecastData,3,length(biomarkers_staging_single)); % disease/model time
t_to_startDate_ForecastData = nan(nIndividuals_ForecastData,length(biomarkers_staging_single)); % time difference from staging date (weighted average across biomarkers) to first forecast
%* Training set
tStageTrainingData_quartiles_all = nan(nIndividuals_TrainingData,3,length(biomarkers_staging_single));
t_to_startDate_TrainingData = nan(nIndividuals_TrainingData,length(biomarkers_staging_single));
for kb=1:length(biomarkers_staging_single)
  if ~isempty(biomarkers_staging_single{kb})
    tStageForecastData_quartiles_all(:,:,kb) = biomarkers_staging_single{kb}.tStageForecastData_quartiles;
    t_to_startDate_ForecastData(:,kb) = datenum(biomarkers_staging_single{kb}.dateStageForecastData_mean(:)) - startDatenum;
    %rowz = ~strcmpi(biomarkers_staging_single{kb}.dateStageForecastData_mean,''); % non-missing data
    %if any(rowz)
    %  t_to_startDate_ForecastData(rowz,kb) = datenum(biomarkers_staging_single{kb}.dateStageForecastData_mean(rowz)) - startDatenum;
    %end
    %* Training data
    tStageTrainingData_quartiles_all(:,:,kb) = biomarkers_staging_single{kb}.tStage_quartiles;
    rowz = ~strcmpi(biomarkers_staging_single{kb}.dateStageTrainingData_mean,'');
    if any(rowz)
      t_to_startDate_TrainingData(rowz,kb) = datenum(biomarkers_staging_single{kb}.dateStageTrainingData_mean(rowz)) - startDatenum;
    end
  end
end
t_to_startDate_ForecastData = t_to_startDate_ForecastData/daysInAYear;
t_to_startDate_TrainingData = t_to_startDate_TrainingData/daysInAYear;
%* Weight staging per biomarker by inverse "confidence"
tStageForecastData_weighted = nan(nIndividuals_ForecastData,1);
t_to_startDate_ForecastData_weighted = nan(nIndividuals_ForecastData,1);
for ki=1:nIndividuals_ForecastData
  tStageForecastData_ki = permute(tStageForecastData_quartiles_all(ki,:,:),[3,2,1]); % Vertical stack of disease time/stage quartiles (50,25,75) for each biomarker
  t_to_startDate_ki = t_to_startDate_ForecastData(ki,:);
  
  iqr_ki = diff(tStageForecastData_ki(:,[2,3]),1,2); % Inter-quartile range
  d = iqr_ki + eps; % +eps avoids divide-by-0
  weights = 1./d; % inverse "confidence"
  tStageForecastData_weighted(ki) = nansum(tStageForecastData_ki(:,1).*weights)./nansum(weights);
  t_to_startDate_ForecastData_weighted(ki) = nansum(t_to_startDate_ki(:).*weights(:)) ./nansum(weights);
end

%** Missing data: replace using tStage(biomarker(age))
missing_weighted_proxyBiomarker = 'ADAS13';
bio = strcmpi(biomarkers,missing_weighted_proxyBiomarker);
%* Linear model: biomarker ~ AGE
missing_model_weighted_AGE = fitlm(table_TrainingData(:,{missing_weighted_proxyBiomarker,'AGE'}),[missing_weighted_proxyBiomarker,' ~ AGE'],'RobustOpts','on');
%* tStage = GP regression(biomarker)
[x_bio,I_bio] = sort(biomarkers_staging_single{bio}.xStage_mean);
t_bio = biomarkers_staging_single{bio}.tStage_quartiles(I_bio,1);
missing_model_t_weighted = fitrgp(x_bio(~isnan(x_bio)),t_bio(~isnan(x_bio)));
%*** Identify and impute missing data:
missing_tStageForecastData_weighted_bool = isnan(tStageForecastData_weighted);
missing_tStageForecastData_weighted_RID = idStageForecastData_u(missing_tStageForecastData_weighted_bool);
missing_tStageForecastData_weighted_imputed = nan(size(missing_tStageForecastData_weighted_RID));
missing_weighted_imputed = nan(size(missing_tStageForecastData_weighted_imputed));
missing_tStageForecastData_weighted_imputed_CI = nan(size(missing_tStageForecastData_weighted_imputed,1),2);
missing_tStageForecastData_weighted_meanAGE = nan(size(missing_tStageForecastData_weighted_RID));
for ki=1:length(missing_tStageForecastData_weighted_RID)
  missing_tStageForecastData_weighted_meanAGE(ki) = nanmean(table_ForecastData.AGE(table_ForecastData.RID==missing_tStageForecastData_weighted_RID(ki)));
  missing_weighted_imputed(ki) = predict(missing_model_weighted_AGE,missing_tStageForecastData_weighted_meanAGE(ki));
  [missing_tStageForecastData_weighted_imputed(ki),~,missing_tStageForecastData_weighted_imputed_CI(ki,:)] = predict(missing_model_t_weighted, missing_weighted_imputed(ki) ,'Alpha',0.5);
end
%figure,plot(t_bio,x_bio,'k.');hold on;plot(missing_tStageLB2_weighted_simple_imputed,missing_weighted_simple_imputed,'r+')
tStageForecastData_weighted_imputed = tStageForecastData_weighted;
tStageForecastData_weighted_imputed(missing_tStageForecastData_weighted_bool) = missing_tStageForecastData_weighted_imputed;
if plotting
  figure
  subplot(121),plot(t_bio,x_bio,'k.');hold on;plot(missing_tStageForecastData_weighted_imputed,missing_weighted_imputed,'r+')
  subplot(122),plot(tStageForecastData_weighted_simple,tStageForecastData_weighted_imputed,'x'),hold all,plot(missing_tStageForecastData_weighted_imputed,tStageForecastData_weighted_imputed(missing_tStageForecastData_weighted_bool),'+')
end
%* Impute
tStageForecastData_weighted = tStageForecastData_weighted_imputed;
if all(~ismember({'D2','LB2'},forecastData))
  t_to_startDate_ForecastData_weighted = nanmean(t_to_startDate_ForecastData,2);
end

%*** b) Simple single-biomarker forecasts - Ventricles/ICV
bio = strcmpi(biomarkers,'Ventricles_ICV');
tStageForecastData_quartiles_Ventricles_ICV = tStageForecastData_quartiles_all(:,:,bio);
t_to_startDate_ForecastData_Ventricles_ICV = t_to_startDate_ForecastData(:,bio);
tStageTrainingData_Ventricles_ICV_simple = tStageTrainingData_quartiles_all(:,1,bio);
tStageForecastData_Ventricles_ICV_simple = tStageForecastData_quartiles_Ventricles_ICV(:,1);
%** Missing data: replace using tStage(biomarker(age))
%* biomarker ~ AGE
missing_model_VentsICV_AGE = fitlm(table_TrainingData(:,{'Ventricles_ICV','AGE'}),'Ventricles_ICV ~ AGE','RobustOpts','on');
%figure,plot(missing_model_VentsICV_AGE)
%* tStage = GP regression(biomarker)
[x_bio,I_bio] = sort(biomarkers_staging_single{bio}.xStage_mean);
t_bio = biomarkers_staging_single{bio}.tStage_quartiles(I_bio,1);
missing_model_t_VentsICV = fitrgp(x_bio(~isnan(x_bio)),t_bio(~isnan(x_bio)));
%*** Identify and impute missing data:
missing_tStageForecastData_Ventricles_ICV_simple_bool = isnan(tStageForecastData_Ventricles_ICV_simple);
missing_tStageForecastData_Ventricles_ICV_simple_RID = idStageForecastData_u(missing_tStageForecastData_Ventricles_ICV_simple_bool);
missing_tStageForecastData_Ventricles_ICV_simple_imputed = nan(size(missing_tStageForecastData_Ventricles_ICV_simple_RID));
missing_Ventricles_ICV_simple_imputed = nan(size(missing_tStageForecastData_Ventricles_ICV_simple_imputed));
missing_tStageForecastData_Ventricles_ICV_simple_imputed_CI = nan(size(missing_tStageForecastData_Ventricles_ICV_simple_imputed,1),2);
missing_tStageForecastData_Ventricles_ICV_simple_meanAGE = nan(size(missing_tStageForecastData_Ventricles_ICV_simple_RID));
for ki=1:length(missing_tStageForecastData_Ventricles_ICV_simple_RID)
  missing_tStageForecastData_Ventricles_ICV_simple_meanAGE(ki) = nanmean(table_ForecastData.AGE(table_ForecastData.RID==missing_tStageForecastData_Ventricles_ICV_simple_RID(ki)));
  missing_Ventricles_ICV_simple_imputed(ki) = predict(missing_model_VentsICV_AGE,missing_tStageForecastData_Ventricles_ICV_simple_meanAGE(ki));
  [missing_tStageForecastData_Ventricles_ICV_simple_imputed(ki),~,missing_tStageForecastData_Ventricles_ICV_simple_imputed_CI(ki,:)] = predict(missing_model_t_VentsICV, missing_Ventricles_ICV_simple_imputed(ki) ,'Alpha',0.5);
end
%figure,plot(t_bio,x_bio,'k.');hold on;plot(missing_tStageLB2_Ventricles_ICV_simple_imputed,missing_Ventricles_ICV_simple_imputed,'r+')
tStageForecastData_Ventricles_ICV_simple_imputed = tStageForecastData_Ventricles_ICV_simple;
tStageForecastData_Ventricles_ICV_simple_imputed(missing_tStageForecastData_Ventricles_ICV_simple_bool) = missing_tStageForecastData_Ventricles_ICV_simple_imputed;
if plotting
  figure
  subplot(121),plot(t_bio,x_bio,'k.');hold on;plot(missing_tStageForecastData_Ventricles_ICV_simple_imputed,missing_Ventricles_ICV_simple_imputed,'r+')
  subplot(122),plot(tStageForecastData_Ventricles_ICV_simple,tStageForecastData_Ventricles_ICV_simple_imputed,'x'),hold all,plot(missing_tStageForecastData_Ventricles_ICV_simple_imputed,tStageForecastData_Ventricles_ICV_simple_imputed(missing_tStageForecastData_Ventricles_ICV_simple_bool),'+')
end

%*** Simple single-biomarker forecasts - ADAS13
bio = strcmpi(biomarkers,'ADAS13');
tStageForecastData_quartiles_ADAS13 = tStageForecastData_quartiles_all(:,:,bio);
tStageForecastData_ADAS13_simple = tStageForecastData_quartiles_ADAS13(:,1);
t_to_startDate_ADAS13 = t_to_startDate_ForecastData(:,bio);
tStageTrainingData_ADAS13_simple = tStageTrainingData_quartiles_all(:,1,bio);
%** Missing data: replace using tStage(biomarker(age))
%* biomarker ~ AGE
missing_model_ADAS13_AGE = fitlm(table_TrainingData(:,{'ADAS13','AGE'}),'ADAS13 ~ AGE','RobustOpts','on');
%figure,plot(missing_model_ADAS13_AGE)
%* tStage = GP regression(biomarker)
[x_bio,I_bio] = sort(biomarkers_staging_single{bio}.xStage_mean);
t_bio = biomarkers_staging_single{bio}.tStage_quartiles(I_bio,1);
missing_model_t_ADAS13 = fitrgp(x_bio(~isnan(x_bio)),t_bio(~isnan(x_bio)));

missing_tStageForecastData_ADAS13_simple_bool = isnan(tStageForecastData_ADAS13_simple);
missing_tStageForecastData_ADAS13_simple_RID = idStageForecastData_u(missing_tStageForecastData_ADAS13_simple_bool);
missing_tStageForecastData_ADAS13_simple_imputed = nan(size(missing_tStageForecastData_ADAS13_simple_RID));
missing_ADAS13_simple_imputed = nan(size(missing_tStageForecastData_ADAS13_simple_imputed));
missing_tStageForecastData_ADAS13_simple_imputed_CI = nan(size(missing_tStageForecastData_ADAS13_simple_imputed,1),2);
missing_tStageForecastData_ADAS13_simple_meanAGE = nan(size(missing_tStageForecastData_ADAS13_simple_RID));
for ki=1:length(missing_tStageForecastData_ADAS13_simple_RID)
  missing_tStageForecastData_ADAS13_simple_meanAGE(ki) = nanmean(table_ForecastData.AGE(table_ForecastData.RID==missing_tStageForecastData_ADAS13_simple_RID(ki)));
  missing_ADAS13_simple_imputed(ki) = predict(missing_model_ADAS13_AGE,missing_tStageForecastData_ADAS13_simple_meanAGE(ki));
  [missing_tStageForecastData_ADAS13_simple_imputed(ki),~,missing_tStageForecastData_ADAS13_simple_imputed_CI(ki,:)] = predict(missing_model_t_ADAS13, missing_ADAS13_simple_imputed(ki) ,'Alpha',0.5);
end
%*** Impute missing data
tStageForecastData_ADAS13_simple_imputed = tStageForecastData_ADAS13_simple;
tStageForecastData_ADAS13_simple_imputed(missing_tStageForecastData_ADAS13_simple_bool) = missing_tStageForecastData_ADAS13_simple_imputed;
if plotting
  figure
  subplot(121),plot(t_bio,x_bio,'k.');hold on;plot(missing_tStageForecastData_ADAS13_simple_imputed,missing_ADAS13_simple_imputed,'r+')
  subplot(122),plot(tStageForecastData_ADAS13_simple,tStageForecastData_ADAS13_simple_imputed,'x'),hold all,plot(missing_tStageForecastData_ADAS13_simple_imputed,tStageForecastData_ADAS13_simple_imputed(missing_tStageForecastData_ADAS13_simple_bool),'+')
end

fprintf('Missing temporal staging data has been imputed for the simple model, using average age across visits.\n')
fprintf('time ~ GP(biomarker), where biomarker ~ AGE \n')
tStageForecastData_Ventricles_ICV_simple = tStageForecastData_Ventricles_ICV_simple_imputed;
tStageForecastData_ADAS13_simple = tStageForecastData_ADAS13_simple_imputed;

%* Training set
tStageTrainingData_weighted = nan(nIndividuals_TrainingData,1);
t_to_startDate_weighted_TrainingData = nan(nIndividuals_TrainingData,1);
for ki=1:nIndividuals_TrainingData
  tStageTrainingData_ki = permute(tStageTrainingData_quartiles_all(ki,:,:),[3,2,1]);
  t_to_startDate_TrainingData_ki = t_to_startDate_TrainingData(ki,:);
  
  d = diff(tStageTrainingData_ki(:,[2,3]),1,2) + eps; % +eps handles 1/0 problem
  weights = 1./d; % inverse "confidence"
  tStageTrainingData_weighted(ki) = nansum(tStageTrainingData_ki(:,1).*weights)./nansum(weights);
  t_to_startDate_weighted_TrainingData(ki) = nansum(t_to_startDate_TrainingData_ki(:).*weights(:)) ./nansum(weights);
end

%* Generate the required number of Forecasts
hdr = {'RID','ForecastMonth', 'ForecastDate', 'CNRelativeProbability', 'MCIRelativeProbability', 'ADRelativeProbability', ...
  'ADAS13', 'ADAS1350_CILower', 'ADAS1350_CIUpper', 'Ventricles_ICV', 'Ventricles_ICV50_CILower', 'Ventricles_ICV50_CIUpper' };
dataTable_forecast = cell2table(cell(nIndividuals_ForecastData*nForecasts,length(hdr)), 'VariableNames', hdr);
%* Repeated matrices - compare with submission template
dataTable_forecast.RID = reshape(repmat(RIDForecastingData_u, [1, nForecasts])', nIndividuals_ForecastData*nForecasts, 1);
dataTable_forecast.ForecastMonth = repmat((1:nForecasts)', [nIndividuals_ForecastData, 1]);
%* Forecast dates
ForecastDate1 = cell(nForecasts,1);
for m=1:nForecasts
  ForecastDate1{m} = datestr(addtodate(startDatenum, m-1, 'month'), 'yyyy-mm');
end
%* Repeated matrices for submission dates - compare with submission template
dataTable_forecast.ForecastDate = repmat(ForecastDate1, [nIndividuals_ForecastData, 1]);

dataTable_forecast_simple = dataTable_forecast; % simple version uses only single biomarkers, instead of weighting across all of them

%****** Forecasting Ventricles/ICV
%*** Staging version 0: weighted across biomarkers
DEM_object = biomarkers_staging_single{strcmpi('Ventricles_ICV',biomarkers)}.DEM_object;
%* Disease stage weighted over all biomarkers
tStageForecastData_ = tStageForecastData_weighted;
t_to_startDate_ = t_to_startDate_ForecastData_weighted;
Ventricles_ICV_forecasts = TADPOLE_Oxtoby_DEM_ForecastContinuous(DEM_object,tStageForecastData_,t_to_startDate_,tForecast_since_startDate,RIDForecastingData_u);
v = vertcat(Ventricles_ICV_forecasts{:});
dataTable_forecast.Ventricles_ICV = v(:,1);
dataTable_forecast.Ventricles_ICV50_CILower = v(:,2);
dataTable_forecast.Ventricles_ICV50_CIUpper = v(:,3);
%*** Staging version 1: Ventricles/ICV alone
tStageForecastData_ = tStageForecastData_Ventricles_ICV_simple;
t_to_startDate_ = t_to_startDate_ForecastData_Ventricles_ICV;
tic
Ventricles_ICV_forecasts_simple = TADPOLE_Oxtoby_DEM_ForecastContinuous(DEM_object,tStageForecastData_,t_to_startDate_,tForecast_since_startDate,RIDForecastingData_u);
toc
v_simple = vertcat(Ventricles_ICV_forecasts_simple{:});
dataTable_forecast_simple.Ventricles_ICV = v_simple(:,1);
dataTable_forecast_simple.Ventricles_ICV50_CILower = v_simple(:,2);
dataTable_forecast_simple.Ventricles_ICV50_CIUpper = v_simple(:,3);

%* ADAS13, rounded to integer values (can't find where I did the rounding)
DEM_object = biomarkers_staging_single{strcmpi('ADAS13',biomarkers)}.DEM_object;
ADAS13_forecasts = TADPOLE_Oxtoby_DEM_ForecastContinuous(DEM_object,tStageForecastData_weighted,t_to_startDate_ForecastData_weighted,tForecast_since_startDate,RIDForecastingData_u);
a = vertcat(ADAS13_forecasts{:});
%* Fix scores beyond the maximum
ADAS13_max = 85;
ADAS13_FixTheseForecasts = any(a>ADAS13_max,2);
a(ADAS13_FixTheseForecasts,3) = ADAS13_max; % Upper bound
a(ADAS13_FixTheseForecasts,2) = nanmax(DEM_object.x); % Lower bound
a(ADAS13_FixTheseForecasts,1) = nanmean(a(ADAS13_FixTheseForecasts,[2,3]),2);
dataTable_forecast.ADAS13 = a(:,1);
dataTable_forecast.ADAS1350_CILower = a(:,2);
dataTable_forecast.ADAS1350_CIUpper = a(:,3);

%* Disease stage based on ADAS13 alone
tStageForecastData_ = tStageForecastData_ADAS13_simple;
t_to_startDate_ = t_to_startDate_ADAS13;
ADAS13_forecasts_simple = TADPOLE_Oxtoby_DEM_ForecastContinuous(DEM_object,tStageForecastData_,t_to_startDate_,tForecast_since_startDate,RIDForecastingData_u);
a_simple = vertcat(ADAS13_forecasts_simple{:});
dataTable_forecast_simple.ADAS13 = a_simple(:,1);
dataTable_forecast_simple.ADAS1350_CILower = a_simple(:,2);
dataTable_forecast_simple.ADAS1350_CIUpper = a_simple(:,3);

%*** Diagnosis probabilities based on training set
%* Rescale the numerical diagnosis:
%  11/21/31 are CN;
%  12/22/32 are MCI;
%  13/23/33 are AD.
%* I map them thusly:
%  11   10    = stable NL
%  21   13.33 = MCI to NL
%  31   16.67 = Dementia to NL
%  12   16.67 = NL  to MCI
%  22   20    = stable MCI
%  32   23.33 = Dementia to MCI
%  13   23.33 = NL  to Dementia
%  23   26.67 = MCI to Dementia
%  33   30    = stable Dementia
fprintf('Forecasting diagnostic probabilities based on training set:\n  Full  : p(DX) = exp(-[DXNUMnew-DX0]^2/22); with DX0 = 10/20/30 for CN/MCI/AD.; and DXNUMnew being weighted between these, e.g., DXNUM = 12 => DXNUMnew = 16.67\n')
fwhm = 25/log(2);
pCN  = @(dx) exp(-abs(dx-10).^2/(fwhm/2));
pMCI = @(dx) exp(-abs(dx-20).^2/fwhm);
pAD  = @(dx) exp(-abs(dx-30).^2/(fwhm/2));

DX_weights_primary_TrainingData   = 10*rem(table_TrainingData.DXNUM,10);
DX_weights_secondary_TrainingData = 10*fix(table_TrainingData.DXNUM/10);
DX_weights_TrainingData = (2*DX_weights_primary_TrainingData + 1*DX_weights_secondary_TrainingData)/3;
%x_DX = linspace(10,30,100); figure,plot(x_DX,pCN(x_DX),'g','LineWidth',4),hold all,plot(x_DX,pMCI(x_DX),'b','LineWidth',4),plot(x_DX,pAD(x_DX),'r','LineWidth',4)
DX_weights_mean_TrainingData = nan(size(idStage_u));
for kID=1:length(idStage_u)
  rowz = table_TrainingData.RID==idStage_u(kID);
  DX_weights_mean_TrainingData(kID) = nanmean(DX_weights_TrainingData(rowz));
end
%* Calculate probability of DX
pCN_i  = pCN(DX_weights_mean_TrainingData);
pMCI_i = pMCI(DX_weights_mean_TrainingData);
pAD_i  = pAD(DX_weights_mean_TrainingData);
%* Calculate probability as smooth function of time via GP regression
[t,I] = sort(tStageTrainingData_weighted);
GPR_model_CN = fitrgp(t,pCN_i(I));
GPR_model_MCI = fitrgp(t,pMCI_i(I));
GPR_model_AD = fitrgp(t,pAD_i(I));
tp = linspace(t(1),t(end),1000);
alf = 0.5;
[pCN_t,~,pCN_t_ci]  = predict(GPR_model_CN, tp(:),'Alpha',alf);
[pMCI_t,~,pMCI_t_ci]  = predict(GPR_model_MCI, tp(:),'Alpha',alf);
[pAD_t,~,pAD_t_ci]  = predict(GPR_model_AD, tp(:),'Alpha',alf);
%figure,errorbar_shadow(tp,pCN_t,pCN_t-pCN_t_ci(:,1),pCN_t_ci(:,2)-pCN_t);
%* Ensure 0 <= p <= 1
pCN_t(pCN_t<0)   = 0; pCN_t(pCN_t>1)   = 1; pCN_t_ci(pCN_t_ci<0) = 0; pCN_t_ci(pCN_t_ci>1) = 1;
pMCI_t(pMCI_t<0) = 0; pMCI_t(pMCI_t>1) = 1; pMCI_t_ci(pMCI_t_ci<0) = 0; pMCI_t_ci(pMCI_t_ci>1) = 1;
pAD_t(pAD_t<0)   = 0; pAD_t(pAD_t>1)   = 1; pAD_t_ci(pAD_t_ci<0) = 0; pAD_t_ci(pAD_t_ci>1) = 1;
%* Normalise
P = sum([pCN_t(:),pMCI_t(:),pAD_t(:)],2);
pRel_CN = pCN_t./P;
pRel_MCI = pMCI_t./P;
pRel_AD = pAD_t./P;
%figure,plot(tp,pRel_CN,'g',tp,pRel_MCI,'b',tp,pRel_AD,'r',tp,sum([pRel_CN(:),pRel_MCI(:),pRel_AD(:)],2),'k')
if plotting
  figure
  plot(t,pCN_i(I),'g.',tp,pCN_t,'g-','LineWidth',4)
  hold all
  plot(t,pMCI_i(I),'b.',tp,pMCI_t,'b-','LineWidth',4)
  plot(t,pAD_i(I),'r.',tp,pAD_t,'R-','LineWidth',4)
  %* 1-alf confidence interval
  plot(tp,pCN_t_ci,'g:')
  plot(tp,pMCI_t_ci,'b:')
  plot(tp,pAD_t_ci,'r:')
  ylabel('Rel Prob')
end

%*** Above was for training set. Now for prediction set ***
%** Full model
%* Disease time corresponding to start date
t0 = tStageForecastData_weighted - t_to_startDate_ForecastData_weighted;
t0 = repmat(t0(:),1,nForecasts).';
t0 = t0(:);
tp = t0 + ((dataTable_forecast.ForecastMonth-1)/12);
CNRelativeProbability = predict(GPR_model_CN,tp(:));
MCIRelativeProbability = predict(GPR_model_MCI,tp(:));
ADRelativeProbability = predict(GPR_model_AD,tp(:));
%* Ensure 0 <= p <= 1
CNRelativeProbability(CNRelativeProbability<0)   = 0;
CNRelativeProbability(CNRelativeProbability>1)   = 1;
MCIRelativeProbability(MCIRelativeProbability<0) = 0;
MCIRelativeProbability(MCIRelativeProbability>1) = 1;
ADRelativeProbability(ADRelativeProbability<0)   = 0;
ADRelativeProbability(ADRelativeProbability>1)   = 1;
%* Normalise
P = sum([CNRelativeProbability(:),MCIRelativeProbability(:),ADRelativeProbability(:)],2);
CNRelativeProbability  = CNRelativeProbability./P;
MCIRelativeProbability = MCIRelativeProbability./P;
ADRelativeProbability  = ADRelativeProbability./P;
%* Save
dataTable_forecast.CNRelativeProbability = CNRelativeProbability;
dataTable_forecast.MCIRelativeProbability = MCIRelativeProbability;
dataTable_forecast.ADRelativeProbability = ADRelativeProbability;
%** Simple model - use ADAS13
%* Disease time corresponding to start date
t0 = tStageForecastData_ADAS13_simple - t_to_startDate_ADAS13;
t0 = repmat(t0(:),1,nForecasts).';
t0 = t0(:);
tp = t0 + ((dataTable_forecast_simple.ForecastMonth-1)/12);
CNRelativeProbability_simple = predict(GPR_model_CN,tp(:));
MCIRelativeProbability_simple = predict(GPR_model_MCI,tp(:));
ADRelativeProbability_simple = predict(GPR_model_AD,tp(:));
%* Ensure 0 <= p <= 1
CNRelativeProbability_simple(CNRelativeProbability_simple<0)   = 0;
CNRelativeProbability_simple(CNRelativeProbability_simple>1)   = 1;
MCIRelativeProbability_simple(MCIRelativeProbability_simple<0) = 0;
MCIRelativeProbability_simple(MCIRelativeProbability_simple>1) = 1;
ADRelativeProbability_simple(ADRelativeProbability_simple<0)   = 0;
ADRelativeProbability_simple(ADRelativeProbability_simple>1)   = 1;
%* Normalise
P_simple = sum([CNRelativeProbability_simple(:),MCIRelativeProbability_simple(:),ADRelativeProbability_simple(:)],2);
CNRelativeProbability_simple  = CNRelativeProbability_simple./P;
MCIRelativeProbability_simple = MCIRelativeProbability_simple./P;
ADRelativeProbability_simple  = ADRelativeProbability_simple./P;
%* Save
dataTable_forecast_simple.CNRelativeProbability = CNRelativeProbability_simple;
dataTable_forecast_simple.MCIRelativeProbability = MCIRelativeProbability_simple;
dataTable_forecast_simple.ADRelativeProbability = ADRelativeProbability_simple;

hdr = 'RID, Forecast Month, Forecast Date, CN relative probability, MCI relative probability, AD relative probability, ADAS13, ADAS13 50% CI lower, ADAS13 50% CI upper, Ventricles_ICV, Ventricles_ICV 50% CI lower, Ventricles_ICV 50% CI upper';
%* Write to CSV
writetable(dataTable_forecast,TADPOLE_Submission_csv,'WriteVariableNames',true)
writetable(dataTable_forecast_simple,TADPOLE_Submission_csv_simple,'WriteVariableNames',true)
%* Replace header
%fid = fopen(TADPOLE_Submission_Leaderboard_csv,'w');
%fwrite(fid,hdr)
%fclose(fid)


%% ***** Some other stuff for the TADPOLE report: extracted manually
runtimes_D1training = [39,1.4*60,43,24,26,48,16,40,3.5,3.5,3.8,19,34,5.2,18];
nDataPoints_D1Training = [341,321,352,302,305,304,172,51,140,140,139,263,337,165,265];

fprintf('Mean (std) run time for model building was %g (%g) minutes.\n',mean(runtimes_D1training),std(runtimes_D1training))

if plotting
  figure
  plot(nDataPoints_D1Training,runtimes_D1training,'+','MarkerSize',15,'LineWidth',2)
  for kb=1:length(biomarkers)
    text(nDataPoints_D1Training(kb),runtimes_D1training(kb),biomarkersLabels{kb})
  end
  ylabel('Model building runtime (minutes)')
  xlabel('Number of data points')
  fnam = [mfile,'__runtimes'];
  export_fig(fnam,'-pdf','-png'); savefig([fnam,'.fig'])
  set(gcf,'Color','white'),box off
end

