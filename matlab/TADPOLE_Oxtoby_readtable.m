function dataTable = TADPOLE_Oxtoby_readtable(dataFile)
%TADPOLE_Oxtoby_readtable Simple wrapper for data read
%
% Modified from DEM_ADNI_readtable
%
% Customised for TADPOLE_D1_D2.csv spreadsheet
%
% Input:  data file path
% Output: MATLAB data table
%
% Author: Neil Oxtoby, UCL, October 2017
% Developed in MATLAB version 8.6.0.267246 (R2015b)

missingDataNumeric = '';

%* Read in the table
dataTable = sortrows(readtable(dataFile,'TreatAsEmpty',missingDataNumeric),{'RID','EXAMDATE'});

dataTable.Properties.Description = 'TADPOLE D1/D2 table created for TADPOLE Challenge 2017.';

