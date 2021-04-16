function varargout = TADPOLE_Oxtoby_DataPrepD3(dataTable_TADPOLE)
%TADPOLE_Oxtoby_DataPrepD3 Simple wrapper for data preparation
%
% Modified from TADPOLE_Oxtoby_DataPrep()
%
% Input:  TADPOLE data table D3
% Output: Modified data table
%    * standardizeMissing()
%    * (not required) ADNI's standard pipeline, per training slides;
%    * select markers;
%    * define normal/abnormal, including reference controls for covariate
%    adjustment
%    * k-fold CV sets, stables/progressors;
%    * handle missing data: remove individuals with missing demographics?
%    * adjust for healthy ageing (covariates);
%
% Author: Neil Oxtoby, UCL, October 2017
% Developed in MATLAB version 8.6.0.267246 (R2015b)

%% *** Missing data
missingDataCodesADNI = {'-4','-2','-1','','NaN',NaN,-4,-2,-1};
dataTable_TADPOLE = standardizeMissing(dataTable_TADPOLE,missingDataCodesADNI);

%* Missing demographics and covariates of interest
%* 1. Age, Gender, Education, Number of APOE4 alleles
%* 2. Study protocol (i.e. ADNI-1, ADNI-GO or ADNI-2)
%* 3. ICV, Scanner field strength, FreeSurfer Version (brain volumes only)
%* NB: Set 1 remain the same over time, Sets 2&3 may change over time
covariatesFixed = {'AGE','PTGENDER','PTEDUCAT','APOE4'};
covariatesDynamic = {'COLPROT'};
covariatesMRI = {'ICV','FLDSTRENG','FSVERSION'};
covs0 = [covariatesFixed,covariatesDynamic];
%* Identify the rows with missing demographic/covariate data
fprintf('Excluding rows with missing: DX, Age, Gender, Education (ideally also ApoE4, but this was not included in D3)\n')
dataTable_missingDemographicsCovariates = strcmpi(dataTable_TADPOLE.DX,'') | isnan(dataTable_TADPOLE.AGE) | strcmpi(dataTable_TADPOLE.PTGENDER,'') | isnan(dataTable_TADPOLE.PTEDUCAT); % | isnan(dataTable_TADPOLE.APOE4);
%   | strcmpi(dataTable_ADNIMERGE.COLPROT,'') ...
%   | strcmpi(dataTable_ADNIMERGE.FLDSTRENG,'') ...
%   | strcmpi(dataTable_ADNIMERGE.FSVERSION,'');
dataTable_excludedRows = dataTable_missingDemographicsCovariates;
dataTable_ = dataTable_TADPOLE;
dataTable_.('MissingDemographicsCovariates') = dataTable_excludedRows;

% dataTable_ = dataTable_(~dataTable_.MissingDemographicsCovariates,:);

%% *** Stables, Progressors, etc. using numeric diagnosis
DXNUM = nan(size(dataTable_.DX));
DX_u = {'NL', 'MCI', 'Dementia', 'NL to MCI', 'NL to Dementia', 'MCI to Dementia', 'MCI to NL', 'Dementia to MCI'};
DXNUM_u = [11,22,33,12,13,23,21,32];
for kdx=1:length(DX_u)
  DXNUM(strcmpi(dataTable_.DX,DX_u{kdx})) = DXNUM_u(kdx);
end
dataTable_.DXNUM = DXNUM;
%* Numeric visit
VISNUM = nan(size(dataTable_.VISCODE));
VISCODE_u = {'bl','m03','m06','m12','m18','m24','m30','m36','m42','m48','m54','m60','m66','m72','m78','m84','m90','m96','m102','m108','m114','m120','m126','m132'};
VISNUM_u = str2double(strrep(strrep(VISCODE_u,'m',''),'bl','0'));
for kvis=1:length(VISNUM_u)
  VISNUM(strcmpi(dataTable_.VISCODE,VISCODE_u{kvis})) = VISNUM_u(kvis);
end
dataTable_.VISNUM = VISNUM;
dataTable_.DXNUM_bl = fillBaseline(DXNUM,dataTable_.RID,VISNUM==0);
%* Numeric gender: male==1
[~,~,PTGENDERNUM] = unique(dataTable_.PTGENDER,'stable');
if PTGENDERNUM(1)==1 && strcmpi(dataTable_.PTGENDER{1},'Male')
  %* Male = 1
  dataTable_.PTGENDERNUM = PTGENDERNUM;
else
  %* Change to Male = 1
  PTGENDERNUM(PTGENDERNUM==1) = PTGENDERNUM(PTGENDERNUM==1)+2;
  dataTable_.PTGENDERNUM = PTGENDERNUM - 1;
end
fprintf('Note: PTGENDER encoded as male = 1, female = 2\n')
clear PTGENDERNUM

% %****** Probably irrelevant ******
% %*** Controls: reference group for calculating correction factors (regression)
% dataTable_.normals = dataTable_.DXNUM==11 & dataTable_.CDRSB==0; %dataTable_.APOE4==0 
% %*** Patients: group upon which to build the DEM
% dataTable_.abnormals = dataTable_.DXNUM==33; % dataTable_.APOE4==1
% %dataTable_.apoe4progressors = progressors==1 & dataTable_.APOE4>0;


fprintf('Data preparation complete (DEM_ADNI_DataPrep.m)\n')
fprintf('  %i individuals remain within %i rows of TADPOLE data\n',length(unique(dataTable_.RID)),size(dataTable_,1))
%* Demographics output 
dataTable_demographics = dataTable_(:,{'RID','VISCODE','DX','DXNUM','AGE','PTEDUCAT','PTGENDER'});
fprintf('  %i Males  (%i%%)\n',sum(strcmpi(dataTable_demographics.PTGENDER,'Male')),round(100*sum(strcmpi(dataTable_demographics.PTGENDER,'Male'))/length(dataTable_demographics.PTGENDER)))


switch nargout
  case 1
    varargout = {dataTable_};
  otherwise
    error('TADPOLE_Oxtoby_DataPrep:nargout','ERROR (TADPOLE_Oxtoby_DataPrep): Incorrect number of outputs.')
end
  

end
