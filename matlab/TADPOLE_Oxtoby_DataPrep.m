function varargout = TADPOLE_Oxtoby_DataPrep(dataTable_TADPOLE)
%TADPOLE_Oxtoby_DataPrep Simple wrapper for data preparation
%
% Modified from DEM_ADNI_DataPrep()
%
% Input:  TADPOLE data table D1_D2
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
%* CSF: could handle weird beyond-assay-detection-threshold exceptions, 
%  but there aren't many: 1/9/12 ABETA/TAU/PTAU
% - PTAU: '<8' and '>120'  => normal and abnormal
% - TAU: '<80' and '>1300' => normal and abnormal
% - ABETA: '<200' => abnormal
%
% By the way, CSF missing (empty string): 
% 10370/10370/10371 ABETA/TAU/PTAU of 12741 rows in D1
%
% => Just convert the strings and let the exceptions become NaN (missing)
dataTable_TADPOLE.ABETA_UPENNBIOMK9_04_19_17 = str2double(dataTable_TADPOLE.ABETA_UPENNBIOMK9_04_19_17);
dataTable_TADPOLE.PTAU_UPENNBIOMK9_04_19_17  = str2double(dataTable_TADPOLE.PTAU_UPENNBIOMK9_04_19_17);
dataTable_TADPOLE.TAU_UPENNBIOMK9_04_19_17   = str2double(dataTable_TADPOLE.TAU_UPENNBIOMK9_04_19_17);

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
fprintf('Excluding rows with missing: DX, Age, Gender, Education, ApoE4\n')
fprintf('\n\n\n****** Converting AGE to AGE + Years_bl (because AGE was actually baseline age in the TADPOLE spreadsheet) ******\n\n\n')
dataTable_TADPOLE.AGE = dataTable_TADPOLE.AGE + dataTable_TADPOLE.Years_bl;
dataTable_missingDemographicsCovariates = strcmpi(dataTable_TADPOLE.DX,'') | isnan(dataTable_TADPOLE.AGE) | strcmpi(dataTable_TADPOLE.PTGENDER,'') | isnan(dataTable_TADPOLE.PTEDUCAT) | isnan(dataTable_TADPOLE.APOE4);
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

%* Stables and Progressors
[ stables, progressors, progressionVisit ] = findStablesAndProgressors( dataTable_.Years_bl, dataTable_.DXNUM, dataTable_.RID );
dataTable_.stables = stables;
dataTable_.progressors = progressors;
dataTable_.progressionVisit = progressionVisit;

%*** Controls: reference group for calculating correction factors (regression)
dataTable_.normals = stables==1 & dataTable_.APOE4==0 & dataTable_.DXNUM==11 & dataTable_.CDRSB==0;

%*** Patients: group upon which to build the DEM
dataTable_.abnormals = stables==1 & dataTable_.APOE4==1 & dataTable_.DXNUM==33;
%dataTable_.apoe4progressors = progressors==1 & dataTable_.APOE4>0;

%% *** 10-fold Cross-validation group
[ADNI_RID_unique,ADNI_RID_unique_I,dataTable_.ID] = unique(dataTable_.RID,'stable');
GENDER = dataTable_.PTGENDERNUM(ADNI_RID_unique_I);
MALE = GENDER==1;
APOE4 = dataTable_.APOE4(ADNI_RID_unique_I);
group1 = MALE==1 & APOE4==1;
group2 = MALE==1 & APOE4==0;
group3 = MALE==0 & APOE4==1;
group4 = MALE==0 & APOE4==0;
groups = 1*group1 + 2*group2 + 3*group3 + 4*group4;
dataTable_CVPartition = cvpartition(groups,'KFold',10);
CVFold = zeros(size(dataTable_CVPartition.test(1)));
for k=1:10
  CVFold = CVFold + k*dataTable_CVPartition.test(k);
end
dataTable_CV = table(ADNI_RID_unique,MALE,APOE4,CVFold,'VariableNames',{'RID','MALE','APOE4','CVFold'});

fprintf('Data preparation complete (DEM_ADNI_DataPrep.m)\n')
fprintf('  %i individuals remain within %i rows of TADPOLE data\n',length(ADNI_RID_unique),size(dataTable_,1))
%* Demographics output 
dataTable_baseline_demographics = dataTable_(strcmpi(dataTable_.VISCODE,'bl'),{'RID','VISCODE','DX','DXNUM','AGE','PTEDUCAT','PTGENDER','APOE4','stables','progressors','normals','abnormals'});
fprintf('  %i Males  (%i%%)\n',sum(strcmpi(dataTable_baseline_demographics.PTGENDER,'Male')),round(100*sum(strcmpi(dataTable_baseline_demographics.PTGENDER,'Male'))/length(dataTable_baseline_demographics.PTGENDER)))
fprintf('  %i APOE4+ (%i%%)\n',sum(dataTable_baseline_demographics.APOE4==1),round(100*sum(dataTable_baseline_demographics.APOE4==1)/length(dataTable_baseline_demographics.PTGENDER)))
fprintf('  %i normals   (APOE4-, stable NL)\n',sum(dataTable_baseline_demographics.normals==1))
fprintf('  %i abnormals (APOE4+, stable AD)\n',sum(dataTable_baseline_demographics.abnormals==1))
fprintf('  %i clinical progressors (%i NL; %i MCI)\n',sum(dataTable_baseline_demographics.progressors==1),sum(dataTable_baseline_demographics.progressors==1 & dataTable_baseline_demographics.DXNUM==11),sum(dataTable_baseline_demographics.progressors==1 & dataTable_baseline_demographics.DXNUM==22))
fprintf('  %i stable MCI\n',sum(dataTable_baseline_demographics.stables==1 & dataTable_baseline_demographics.DXNUM==22))


switch nargout
  case 1
    varargout = {dataTable_};
  case 2
    varargout = {dataTable_,dataTable_CV};
  otherwise
    error('TADPOLE_Oxtoby_DataPrep:nargout','ERROR (TADPOLE_Oxtoby_DataPrep): Incorrect number of outputs.')
end
  

end


function [ stablesAll, progressorsAll , varargout ] = findStablesAndProgressors( t, diagnosis_numerical, id )
%[stables,progressors,progressionVisit] = findStablesAndProgressors(t,d,id)
%  Loops through time points t and monotonically-increasing numerical
%  diagnoses d to find subjects who progress clinically (and those who
%  don't).
%
% Neil Oxtoby, UCL, Dec 2015

%* Unique id
id_uu = unique(id);
%* Progressors, Reverters, Stables, Mixed
progressors = nan(size(id_uu));
progressionVisit = nan(size(id_uu));
progressorsAll = nan(size(id));
progressionVisitAll = false(size(id));

reverters = progressors;
reversionVisit = progressionVisit;
revertersAll = progressorsAll;
reversionVisitAll = progressionVisitAll;

stables = progressors;
stablesAll = progressorsAll;
mixeds = progressors;
mixedsAll = progressorsAll;

nVisits = stables;
nVisitsAll = stablesAll;

%* Loop through id and identify subjects who progress in diagnosis
missingDX = isnan(diagnosis_numerical);
missingT =  isnan(t);
for k = 1 : length(id_uu)
  rowz = id==id_uu(k);
  
  %* Remove missing data
  rowz = rowz & ~missingDX & ~missingT;
  
  %ex = x(rowz);
  tee = t(rowz);
  dee = diagnosis_numerical(rowz);
  [~,ordr] = sort(tee);
  dee = dee(ordr);
  if length(dee)>1
    dee_diff = diff(dee);
    if all(dee_diff>=0)
      if any(dee_diff>0) %* Progressors
        progressors(k) = true;
        stables(k) = false;
        mixeds(k) = false;
        reverters(k) = false;
        %* Identify progression visit
        progressionVisit(k) = find(dee>dee(1),1,'first');
        rowz_f = find(rowz);
        progressionVisitAll(rowz_f(progressionVisit(k))) = true;
      else %* Stables
        stables(k) = true;
        progressors(k) = false;
        mixeds(k) = false;
        reverters(k) = false;
      end
    elseif all(dee_diff<=0)
      if any(dee_diff<0) %* Reverters
        reverters(k) = true;
        progressors(k) = false;
        stables(k) = false;
        mixeds(k) = false;
        %* Identify reversion visit
        reversionVisit(k) = find(dee<dee(1),1,'first');
        rowz_f = find(rowz);
        reversionVisitAll(rowz_f(reversionVisit(k))) = true;
      end
    else %* Mixed
      mixeds(k) = true;
      progressors(k) = false;
      stables(k) = false;
      reverters(k) = false;
    end
  end
  nVisits(k) = sum(~isnan(dee) & ~isnan(tee));
  progressorsAll(rowz) = progressors(k);
  stablesAll(rowz) = stables(k);
  mixedsAll(rowz) = mixeds(k);
  revertersAll(rowz) = reverters(k);
  nVisitsAll(rowz) = nVisits(k);
end

if nargout>2
  varargout{1} = progressionVisitAll;
  varargout{2} = nVisitsAll;
  varargout{3} = revertersAll;
  varargout{4} = reversionVisitAll;
  varargout{5} = mixedsAll;
end

end