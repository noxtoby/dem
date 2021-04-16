function biomarkerForecasts = TADPOLE_Oxtoby_DEM_ForecastContinuous(DEM_object,t_diseaseStage,t_to_startDate,tForecast_since_startDate,RID_u)
%TADPOLE_Oxtoby_DEM_ForecastContinuous Produces forecasts for continuous target variables
%
% Developed for TADPOLE Challenge 2017.
%
%
% Neil P. Oxtoby, UCL, November 2017

plotting = false;

n = length(t_diseaseStage);
biomarkerForecasts = cell(n,1);
nForecasts = length(tForecast_since_startDate);

xf = DEM_object.x_fit;
tf = DEM_object.t_fit;
xs = DEM_object.x_fit_samples;
ts = DEM_object.t_fit_samples;

if plotting; figure; end

%* In case the forecasts go beyond the fit, recalculate the forecast using
%  linear extrapolation, with simple CIs given by 5% of the value
forecastsToRecalculate = cell(n,1);
extrapolateData = (length(tf)-round(length(tf)/5)):length(tf);
extrapolateModel = fitlm(tf(extrapolateData),xf(extrapolateData));

for ki=1:n
  xForecasts = nan(nForecasts,3);
  
  %* disease time corresponding to first forecast (start date)
  t0 = t_diseaseStage(ki);
  tfore_0 = t0 - t_to_startDate(ki);
  tfore_vec = tfore_0 + tForecast_since_startDate(:);
  
  %* Trajectory anchoring
  dt = abs(t0-tf);
  anchor = abs(dt)==min(abs(dt));
  if find(anchor)==length(tf)
    %* Disease stage is beyond the fit, so re-anchor to extrapolation model
    dtfore = t0-tfore_vec;
    anchor = abs(dtfore)==min(abs(dtfore));
    xfore_0 = predict(extrapolateModel,tfore_0);
    xForecasts(:,1) = predict(extrapolateModel,tfore_vec(:));
    xForecasts(:,2) = xForecasts(:,1)*0.95;
    xForecasts(:,3) = xForecasts(:,1)*1.05;
  else
    %* Disease stage is within the fit, so anchor the posterior samples
    %* Closest matching biomarker value
    x0 = xf(dt==min(dt));
    xa = xf(anchor);
    ta = ts;
    ta_upperBound = 100;
    for ks=1:length(ta)
      % Anchor time t=0 => shift time relative to anchor, robustly
      [~,k_] = min(abs(xs{ks}-xa));
      %* Shift time
      ta{ks} = ta{ks} - ts{ks}(k_) + t0;
      ta_upperBound = min(ta_upperBound,max(ta{ks}));
    end
    
    %*** Forecasting
    tForeWithinFit = tfore_vec < ta_upperBound;
    %* Beyond the fit
    xForecasts(not(tForeWithinFit),1) = predict(extrapolateModel,tfore_vec(not(tForeWithinFit)));
    xForecasts(not(tForeWithinFit),2) = 0.95*xForecasts(not(tForeWithinFit),1);
    xForecasts(not(tForeWithinFit),3) = 1.05*xForecasts(not(tForeWithinFit),1);
    %* Within the fit
    %  Scan through the forecast times,
    %    match the biomarker trajectory samples,
    %      calculate the forecasts as the quartiles:
    %  median, 25th percentile, 75th percentile
    if any(tForeWithinFit)
      tForeWithinFit = find(tForeWithinFit);
      for i_kfore=1:length(tForeWithinFit)
        kfore = tForeWithinFit(i_kfore);
        tfore = tForecast_since_startDate(kfore) + tfore_0;
        xmatches = nan(1,length(ta));
        bm = false;
        for ks=1:length(ta)
          tas = ta{ks}; % time for current sample trajectory
          xas = xs{ks};
          dt = abs(tas - tfore);
          bestMatch = dt==min(dt); % best match
          %if find(bestMatch,1,'first')==length(bestMatch) || find(bestMatch,1,'last')==1
          %  bm = true; % flag when the best match is at the end of the data
          %end
          xmatches(ks) = xas(bestMatch);
        end
        %if bm
        %  fprintf('%i,',kfore);
        %  forecastsToRecalculate{ki} = [forecastsToRecalculate{ki},kfore];
        %end
        xForecasts(kfore,:) = quantile(xmatches,[0.50,0.25,0.75]);
      end
    end
    
    xfore_0 = xForecasts(1,1);
    
  end
  
  biomarkerForecasts{ki} = xForecasts;
  
  if plotting
    for ks=1:length(ta)
      plot(ta{ks},xs{ks},'-','Color',0.8*[1,1,1]),hold on
    end
    plot(t0,xa,'b*','MarkerSize',10)
    plot(tfore_0,xfore_0,'kd','MarkerSize',10)
    plot(tf,xf,'r-','LineWidth',2)
    errorbar(tfore_0+tForecast_since_startDate,xForecasts(:,1),xForecasts(:,1)-xForecasts(:,2),xForecasts(:,3)-xForecasts(:,1))
    %legend('Fit','Individual')
    title(sprintf('PAUSED!   RID = %i',RID_u(ki)))
    drawnow
    pause
    hold off
  end
  
end


end



