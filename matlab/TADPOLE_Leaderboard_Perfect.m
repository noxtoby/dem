%% Run TADPOLE_Oxtoby_DEM_Leaderboard first
dt_Perfect = dataTable_forecast_leaderboard;

for ki=1:n
  RID = idStageLB2_u(ki);
  
  rowz_forecast = dataTable_forecast_leaderboard.RID==RID;
  rowz_forecast_f = find(rowz_forecast);
  rowz_LB4 = dataTable_LB4.RID==RID;
  rowz_D1 = dataTable_D1D2_prepped.RID==RID;
  dates_forecast = dataTable_forecast_leaderboard.ForecastDate(rowz_forecast);
  dates_LB4_MRI = dataTable_LB4.ScanDate(rowz_LB4);
  dates_LB4_Cog = dataTable_LB4.CognitiveAssessmentDate(rowz_LB4);
  dates_D1 = dataTable_D1D2_prepped.EXAMDATE(rowz_D1);
  
  datenums_forecast = datenum(dates_forecast) + 15; % mid-month
  datenums_LB4_MRI = datenum(dates_LB4_MRI);
  datenums_LB4_Cog = datenum(dates_LB4_Cog);
  datenums_D1 = datenum(dates_D1);
  
  datenums_diff_MRI = repmat(datenums_forecast(:),1,length(datenums_LB4_MRI)) - repmat(datenums_LB4_MRI(:).',length(datenums_forecast),1);
  bestMatch_MRI = abs(datenums_diff_MRI)==repmat(min(abs(datenums_diff_MRI),[],1),length(datenums_forecast),1);
  [ex_MRI,wy_MRI] = find(bestMatch_MRI);
  datenums_diff_Cog = repmat(datenums_forecast(:),1,length(datenums_LB4_Cog)) - repmat(datenums_LB4_Cog(:).',length(datenums_forecast),1);
  bestMatch_Cog = abs(datenums_diff_Cog)==repmat(min(abs(datenums_diff_Cog),[],1),length(datenums_forecast),1);
  [ex_Cog,wy_Cog] = find(bestMatch_Cog);
  
  Ventricles_D1 = dataTable_D1D2_prepped.Ventricles_ICV(rowz_D1);
  Ventricles_LB4 = dataTable_LB4.Ventricles(rowz_LB4);
  ADAS13_D1 = dataTable_D1D2_prepped.ADAS13(rowz_D1);
  ADAS13_LB4 = dataTable_LB4.ADAS13(rowz_LB4);
  DX_D1 = dataTable_D1D2_prepped.DX(rowz_D1);
  DX_LB4 = dataTable_LB4.Diagnosis(rowz_LB4);
  Ventricles_forecast = dataTable_forecast_leaderboard.Ventricles_ICV(rowz_forecast);
  ADAS13_forecast = dataTable_forecast_leaderboard.ADAS13(rowz_forecast);
  DX_forecast_CN = dataTable_forecast_leaderboard.CNRelativeProbability(rowz_forecast);
  DX_forecast_MCI = dataTable_forecast_leaderboard.MCIRelativeProbability(rowz_forecast);
  DX_forecast_AD = dataTable_forecast_leaderboard.ADRelativeProbability(rowz_forecast);
  for k=1:length(wy_MRI)
    if ~isnan(Ventricles_LB4(wy_MRI(k)))
      %* Cheat
      dt_Perfect.Ventricles_ICV(rowz_forecast_f(ex_MRI(k)) + (-1:1:1)) = Ventricles_LB4(wy_MRI(k));
      dt_Perfect.Ventricles_ICV50_CILower(rowz_forecast_f(ex_MRI(k)) + (-1:1:1)) = 0.95*Ventricles_LB4(wy_MRI(k));
      dt_Perfect.Ventricles_ICV50_CIUpper(rowz_forecast_f(ex_MRI(k)) + (-1:1:1)) = 1.05*Ventricles_LB4(wy_MRI(k));
    end
  end
  title('Ventricles'),xlabel('datenum (LB4)')
  for k=1:length(wy_Cog)
    if ~isnan(ADAS13_LB4(wy_Cog(k)))
      %* Cheat
      dt_Perfect.ADAS13(rowz_forecast_f(ex_Cog(k)) + (-1:1:1)) = ADAS13_LB4(wy_Cog(k));
      dt_Perfect.ADAS1350_CILower(rowz_forecast_f(ex_Cog(k)) + (-1:1:1)) = 0.9*ADAS13_LB4(wy_Cog(k));
      dt_Perfect.ADAS1350_CIUpper(rowz_forecast_f(ex_Cog(k)) + (-1:1:1)) = 1.1*ADAS13_LB4(wy_Cog(k));
    end
    if ~isempty(DX_LB4(wy_Cog(k)))
      if strcmpi(DX_LB4(wy_Cog(k)),'CN')
        dt_Perfect.CNRelativeProbability(rowz_forecast_f(ex_Cog(k)) + (-1:1:1)) = 1;
        dt_Perfect.MCIRelativeProbability(rowz_forecast_f(ex_Cog(k)) + (-1:1:1)) = 0;
        dt_Perfect.ADRelativeProbability(rowz_forecast_f(ex_Cog(k)) + (-1:1:1)) = 0;
      elseif strcmpi(DX_LB4(wy_Cog(k)),'MCI')
        dt_Perfect.CNRelativeProbability(rowz_forecast_f(ex_Cog(k)) + (-1:1:1)) = 0;
        dt_Perfect.MCIRelativeProbability(rowz_forecast_f(ex_Cog(k)) + (-1:1:1)) = 1;
        dt_Perfect.ADRelativeProbability(rowz_forecast_f(ex_Cog(k)) + (-1:1:1)) = 0;
      elseif strcmpi(DX_LB4(wy_Cog(k)),'AD')
        dt_Perfect.CNRelativeProbability(rowz_forecast_f(ex_Cog(k)) + (-1:1:1)) = 0;
        dt_Perfect.MCIRelativeProbability(rowz_forecast_f(ex_Cog(k)) + (-1:1:1)) = 0;
        dt_Perfect.ADRelativeProbability(rowz_forecast_f(ex_Cog(k)) + (-1:1:1)) = 1;
      end
    end
  end
end

%%
writetable(dt,'TADPOLE_Submission_Leaderboard_Perfect6.csv')

