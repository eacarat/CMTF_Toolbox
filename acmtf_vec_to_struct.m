function A = acmtf_vec_to_struct(x, Z, R)
% ACMTF_VEC_TO_STRUCT Converts a vector to a struct which
% contains a cell array of factor matrices and a cell array of norms for 
% each data set
% 
% A = ACMTF_VEC_STRUCT(x, Z, R)
%
% Input:  x: a vector of P, where for an I by J by K tensor and I by L
%            matrix, P = (I+J+K+L+2)*R and R is the number of components.
%         Z: a struct with object, modes, size, miss fields storing the 
%            coupled data sets (See cmtf_check)
%         R: number of components
% 
% Output: A: a struct with fac (a cell array corresponding to factor 
%            matrices) and norms (a cell array corresponding to the weights
%            of components in each data set) fields. 
%
% See also ACMTF_FUN, ACMTF_STRUCT_TO_VEC, CMTF_CHECK.
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
sz  = Z.size;
N   = length(sz);
P   = length(Z.object);

%% Create A
% factor matrices will be the first sum(sz)*R entries
A.fac = cell(N,1);
for n = 1:N
    idx1 = sum(sz(1:n-1))*R + 1;
    idx2 = sum(sz(1:n))*R;
    A.fac{n} = reshape(x(idx1:idx2),sz(n),R);
end
U = R*sum(sz(1:n));
A.norms =cell(P,1);
for p=1:P
    idx1 = U + (p-1)*R+1;
    idx2 = U+  p*R;  
    A.norms{p} = x(idx1:idx2);
end
