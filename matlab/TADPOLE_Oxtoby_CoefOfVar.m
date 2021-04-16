function x_cv = TADPOLE_Oxtoby_CoefOfVar(x,vis,id)
%x_cv = TADPOLE_Oxtoby_CoefOfVar(x,vis,id)
%   Calculate coefficient of variation within-subject (longitudinal scalar biomarker data).
%
%  x   = data (e.g., biomarker values)
%  vis = visit numbers
%  id  = individual ID
%
% Neil Oxtoby, UCL, Nov 2017 

x_matrix = convertVectorsToMatrix(x,vis,id);
id_matrix = convertVectorsToMatrix(id,vis,id);

% [~,~,vis2] = unique(vis,'stable'); % Make sure no zeros

x_cv_vector = nanstd(x_matrix,1)./nanmean(x_matrix,1);
id_cv_vector = nanmean(id_matrix,1);

x_cv = nan(size(x));
% for k=1:length(x_cv)
%   if ~isnan(x(k)) && ~isnan(vis(k)) && ~isnan(id(k))
%     x_cv(k) = x_cv_vector(id(k));
%   end
% end
for ki=1:length(id_cv_vector)
  rowz = id == id_cv_vector(ki);
  cv = x_cv_vector(ki);
  n = ~isnan(x(rowz)) & ~isnan(vis(rowz)) & ~isnan(id(rowz));
  rowz = find(rowz);
  x_cv(rowz(n)) = cv;
end

end

