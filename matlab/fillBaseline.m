function x_bl = fillBaseline(x,id,bl_bool)
% x_bl = fillBaseline(x,id,bl_bool)
%
% x - data (individuals/visits)
% id - individual ID
% bl_bool - boolean vector: true where visit is baseline
%
% Neil Oxtoby, UCL, Nov 2015

x_bl = nan(size(x));
[~,~,idx] = unique(id,'stable');
id_u = unique(idx);
ct = 0;
for ki=1:length(id_u)
  rowz = idx==id_u(ki);
  rowz_bl = rowz & bl_bool;
  if sum(rowz_bl)~=1
    ct = ct+1;
    %fprintf('ID = %i.  %i bl visits.\n',id_u(ki),sum(rowz_bl))
  else
    x_bl(rowz) = x(rowz_bl);
  end
end

if ct>0
  fprintf('%s - Total rows with no bl visit: %i of %i.\n',mfilename,ct,length(x))
end

end