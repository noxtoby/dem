function DEM_object = TADPOLE_Oxtoby_DEM_AnchorTrajectory(DEM_object,x_anchor)
%TADPOLE_Oxtoby_DEM_AnchorTrajectory(DEM_object,x_anchor)
%
% Shifts estimated time to a desired "anchor" point (biomarker value).
%
% Neil Oxtoby, UCL, Oct 2017

% Anchor time t=0 => shift time relative to anchor, robustly
[~,k_] = min(abs(DEM_object.x_fit-x_anchor));
%* Shift time
DEM_object.t_fit = DEM_object.t_fit - DEM_object.t_fit(k_);

for ks=1:length(DEM_object.t_fit_samples)
  % Anchor time t=0 => shift time relative to anchor, robustly
  [~,k_] = min(abs(DEM_object.x_fit_samples{ks}-x_anchor));
  %* Shift time
  DEM_object.t_fit_samples{ks} = DEM_object.t_fit_samples{ks} - DEM_object.t_fit_samples{ks}(k_);
end

end