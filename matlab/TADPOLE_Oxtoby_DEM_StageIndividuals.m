function [tStage,tStageStd,tStage_quartiles,xStage]  = TADPOLE_Oxtoby_DEM_StageIndividuals(x,id,DEM_object)
%TADPOLE_Oxtoby_DEM_StageIndividuals(x,id,DEM_object)
% 
% Aligns the fit from a DEM object to individuals' data points.
% 
% x - data
% id - ID for individuals
% DEM - DEM object with existing fit
%
% For multiple datapoints for any individual idk, the staging uses
% the average value of x(id==idk)
%
% Neil Oxtoby, UCL, Oct 2017

if isempty(DEM_object.x_fit)
  error('stageIndividuals:unfit','DEM object has empty x_fit. Try calling DEM_object.fitDEM() first.')
end
xf = DEM_object.x_fit;
tf = DEM_object.t_fit;
[~,id_2,id_num] = unique(id,'stable'); % numeric ID
id_num_u = id_num(id_2); % numeric unique ID
tStage = nan(size(x));
xStage = nan(length(id_num_u),1);
for ki=1:length(id_num_u)
  rowz = id_num_u(ki)==id_num;
  %* Match average value to closest fit value
  xi = nanmean(x(rowz));
  [~,k_] = min(abs(xi-xf));
  if xi>=nanmin(xf) && xi<=nanmax(xf)
    %* Time from the fit
    tStage(rowz) = tf(k_);
    xStage(ki) = xi;
  end
end

%* Uncertainty (std) in staging, from samples
if nargout>1
  tStage_quartiles = nan(length(id_num_u),3); % Gives 50% coverage probability
  tStageStd = nan(size(tStage));
  tfs = DEM_object.t_fit_samples;
  xfs = DEM_object.x_fit_samples;
  for ki=1:length(id_num_u)
    rowz = id_num_u(ki)==id_num;
    %* Match average value to closest fit value(s)
    xi = nanmean(x(rowz));
    tStage_s = nan(size(tfs));
    for ks=1:length(xfs)
      if not(isempty(xfs{ks}))
        [~,k_] = min(abs(xi-xfs{ks}));
        if xi>=nanmin(xfs{ks}) && xi<=nanmax(xfs{ks})
          %* Time from the fit
          tStage_s(ks) = tfs{ks}(k_);
        end
      end
    end
    tStage_quartiles(ki,:) = quantile(tStage_s,[.50 .25 .75]);
    tStageStd(rowz) = nanstd(tStage_s);
  end
end

end
