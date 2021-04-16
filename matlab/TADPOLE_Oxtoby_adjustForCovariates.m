function [dataTable_adj,dataTable_adjustments_LM] = TADPOLE_Oxtoby_adjustForCovariates(dataTable,referenceGroup,covariates,covariatesMRI,biomarkers,biomarkersMRI)
%TADPOLE_Oxtoby_adjustForCovariatesAdjust for covariates using reference group
% 
% Modified from DEM_ADNI_adjustForCovariates()
% 
% Dependencies:
%   fitlm, ordinal, nanmean
%
% Author: Neil Oxtoby, UCL, October 2017
% Developed in MATLAB version 8.6.0.267246 (R2015b)

%* Convert covariates to ordinal
for kc=1:length(covariates)
  if length(unique(dataTable.(covariates{kc})))<=3
    dataTable.(covariates{kc}) = categorical(dataTable.(covariates{kc}));
  end
end

dataTable_ref = dataTable(referenceGroup,:);
%* Linear regression of reference group
covs_ = strcat(covariates,' + ');
covs_MRI = strcat(covariatesMRI,' + ');
dataTable_adjustments_LM = cell(length(biomarkers),1);
for kb=1:length(biomarkers)
  b = biomarkers{kb};
  if ismember(b,biomarkersMRI)
    covs = [covs_(:).',covs_MRI(:).'];
  else
    covs = covs_(:).';
  end
  lm_func = sprintf('%s ~ 1 + %s%s',b,strcat(covs{1:(end-1)}),covs{end}(1:(end-1)));
  LM = fitlm(dataTable_ref,lm_func);
  dataTable_adjustments_LM{kb} = LM;
  %* Mean-centred regressee
  x_raw = dataTable.(b);
  x_adj = x_raw - ( predict(LM,dataTable) - nanmean(predict(LM,dataTable)) );
  %* Standardized "c-score" (standardized z-score using Controls)
  c_reference = referenceGroup;
  c_mean = nanmean(x_adj(c_reference));
  c_std = nanstd(x_adj(c_reference));
  c_adj = (x_adj - c_mean)/c_std;
  dataTable.([b,'_adj']) = x_adj;
  dataTable.([b,'_adj_cscore']) = c_adj;
end

dataTable_adj = dataTable;

end

