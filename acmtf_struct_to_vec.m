function x = acmtf_struct_to_vec(A)
% ACMTF_STRUCT_TO_VEC Converts a set of factor matrices to a vector.
%
%   X = ACMTF_STRUCT_TO_VEC(A) converts a struct consisting of A.fac cell 
%   array and A.norms cell array to a vector. A.norms are stacked at the end of the vector.
%
% See also ACMTF_FUN, ACMTF_STRUCT_TO_VEC.
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

%% Set-up
N = length(A.fac);
P = length(A.norms);
x = [];

%% Vectorize factor matrices
for n=1:N
    x = [x; A.fac{n}(:)];
end

%% Stack at the end the norms
for p=1:P
    x =[x;A.norms{p}];
end

