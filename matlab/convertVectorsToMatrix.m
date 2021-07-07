function [dataMatrix,indexVectors] = convertVectorsToMatrix( dataVector, dataVectorIndexRow , dataVectorIndexCol )
%convertVectorsToMatrix Converts a dataVector to a matrix indexed by two
% other vectors (say ID and time)
%   
% Usage: 
%   [dataMatrix,indexVectors] = convertVectorsToMatrix( dataVector, dataVectorIndexRow , dataVectorIndexCol )
% where
%   horzcat(dataVectorIndexRow,dataVectorIndexCol,dataVector) is a matrix with rows
%   containing indices (1, 2) for the data (Vector)
%
% Output matrix size is given by the number of unique indices.
%
% Errors not handled.  Missing data assumed to be encoded as NaN.
%
% Neil  Oxtoby, UCL, November 2014

% dataVector = dataVector(:);
% dataVectorIndexRow = dataVectorIndexRow(:);
% dataVectorIndexCol = dataVectorIndexCol(:);

if not( all( size(dataVector)==size(dataVectorIndexRow) ) ) ...
    || not( all( size(dataVector)==size(dataVectorIndexCol) ) ) ...
    || not( all( size(dataVectorIndexRow)==size(dataVectorIndexCol) ) )
  error('Oxtoby:convertVectorsToMatrix:vectorSize','Inputs must be the same size. See help convertVectorsToMatrix for usage.')
end

%* Unique indices
uniqueIndexRow = unique(dataVectorIndexRow(:));
  uniqueIndexRow(isnan(uniqueIndexRow)) = [];
uniqueIndexCol = unique(dataVectorIndexCol(:));
  uniqueIndexCol(isnan(uniqueIndexCol)) = [];
indexVectors = {uniqueIndexRow,uniqueIndexCol};

%* Form the data matrix
dataMatrix = nan(length(uniqueIndexRow),length(uniqueIndexCol));
for k=1:length(dataVector)
  row = ismember(uniqueIndexRow,dataVectorIndexRow(k));
  col = ismember(uniqueIndexCol,dataVectorIndexCol(k));
  dataMatrix(row,col)= dataVector(k);
end
if nargout==1
  clear indexVectors
end

end

