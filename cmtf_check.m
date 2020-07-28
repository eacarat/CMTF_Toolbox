function tf = cmtf_check(Z,A)
%CMTF_CHECK validates a coupled data structure.
%
%   TF = CMTF_CHECK(Z) validates Z as a coupled data structure, meaning
%   that all of its objects are consistent in size. 
%
%   TF = CMTF_CHECK(Z,A) further validates that the set of factor matices
%   defined by A is consistent with Z.
%
% See also CMTF_OPT, ACMTF_OPT
%
% This is the MATLAB CMTF Toolbox.
% References: 
%    - (CMTF) E. Acar, T. G. Kolda, and D. M. Dunlavy, All-at-once Optimization for Coupled
%      Matrix and Tensor Factorizations, KDD Workshop on Mining and Learning
%      with Graphs, 2011 (arXiv:1105.3422v1)
%    - (ACMTF)E. Acar, A. J. Lawaetz, M. A. Rasmussen,and R. Bro, Structure-Revealing Data 
%      Fusion Model with Applications in Metabolomics, IEEE EMBC, pages 6023-6026, 2013.
%    - (ACMTF)E. Acar,  E. E. Papalexakis, G. Gurdeniz, M. Rasmussen, A. J. Lawaetz, M. Nilsson, and R. Bro, 
%      Structure-Revealing Data Fusion, BMC Bioinformatics, 15: 239, 2014.        
%

%% Check that Z is a structure
if ~isstruct(Z)
    error('Z must be a structure');
end

%% Check that Z has the right fields
fieldnames = {'size', 'object', 'modes'};
for i = 1:length(fieldnames)
    if ~isfield(Z,fieldnames{i})
        error('Z must have a field called''%s''',fieldnames{i});
    end
end

%% Check objects
if ~iscell(Z.object)
    error('Z.object must be a cell array');
end

N = numel(Z.object);
if N < 1
    error('Z.object must have at least one object')
end

for i = 1:N
    if ~isa(Z.object{i},'tensor') &&  ~isa(Z.object{i},'sptensor') && ~(ndims(Z.object{i}) == 2) 
        error('Z.object{%d} must be a tensor or a matrix',i);
    end
end

%% Check modes
if ~iscell(Z.modes)
    error('Z.modes must be a cell array');
end

if numel(Z.modes) ~= numel(Z.object)
    error('Z.modes must have the same number of entries as Z.object');
end

%% Check array of sizes and that all sizes are used somewhere
M = numel(Z.size);
SizeUsed = false(M);

for i = 1:N
    if ndims(Z.object{i}) ~= length(Z.modes{i})
        error('Number of modes specified does not match for object %d',i)
    end
    for j = 1:length(Z.modes{i})
        k = Z.modes{i}(j);
        SizeUsed(k) = true;
        if Z.size(k) ~= size(Z.object{i},j)
            error('Size mismatch for object %d',i);
        end       
    end
end
    
for i = 1:M
    if SizeUsed(i) == false
        error('Mode %d is never used in the coupled object', i)
    end
end

%% Exit if there are no factor matrices
if ~exist('A','var')
    tf = true;
    return;
end

%% Check factor matrices

% Check type
if ~iscell(A) && ~isa(A,'ktensor')
    error('A must be a cell array or a ktensor');
end

% Convert to cell array if necessary
if isa(A,'ktensor')
    A = A.u;
end

% Check number of factors
if numel(A) ~= M
    error('There should be %d entries in A', M);
end

% Extract R
R = size(A{1},2);

% Check each matrix size
for i = 1:M
    if ~isequal(size(A{i}), [Z.size(i) R])
        error('Wrong size of factor %d', i);
    end        
end
    
%% Success!
tf = true;


