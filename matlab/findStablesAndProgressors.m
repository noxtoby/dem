function [ stablesAll, progressorsAll , varargout ] = findStablesAndProgressors( t, diagnosis_numerical, id )
%[stables,progressors] = findStablesAndProgressors(t,d,id)
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
for k = 1 : length(id_uu)
  rowz = id==id_uu(k);
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

