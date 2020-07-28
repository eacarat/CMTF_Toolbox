function A = cmtf_vec_to_fac(x,Z)
% CMTF_VEC_TO_FAC Converts a vector to a cell array of factor matrices
%
% A = CMTF_VEC_TO_FAC(x,Z) converts the vector x into a cell array of
% factor matrices consistent with the coupled objects stored in Z.  
%
% Input:  x: a vector of length P, where for an I by J by K tensor and I
%            by L matrix, P = (I+J+K+L)*R and R is the number of
%            components. 
%         Z: a struct with object, modes, size, miss fields storing the 
%            coupled data sets (See cmtf_check)
%
% Output: A: a cell array of factor matrices
%
% See also CMTF_FUN, CMTF_OPT, CMTF_CHECK
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
P   = length(x);
sz  = Z.size;
N   = length(sz);

%% Determine R
R = P / sum(sz);

%% Create A
A = cell(N,1);
for n = 1:N
    idx1 = sum(sz(1:n-1))*R + 1;
    idx2 = sum(sz(1:n))*R;
    A{n} = reshape(x(idx1:idx2),sz(n),R);
end