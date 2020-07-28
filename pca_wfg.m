function [f,G] = pca_wfg(Z,W,A,normZsqr)
% PCA_WFG Function and gradient of matrix factorization with missing data.
%
% [f,G] = pca_wfg(Z,W,A,normZsqr)
%
% Input:  Z: data matrix to be factorized using matrix factorization.
%            Missing entries are replaced with 0.
%         W: binary indicator matrix (same size as Z) containing zeros 
%            wherever there are missing entries in Z (before replacement 
%            by 0).
%         A: a cell array of two factor matrices
%         normZsqr: squared Frobenius norm of Z
%
% Output: f: function value, i.e., f = (1/2) ||W.*(Z - A{1}*A{2}')||^2
%         G: a cell array of two matrices corresponding to the gradients
%
% See also CMTF_FG, PCA_FG
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

%% Compute B = W.*ktensor(A)
%B = W.*full(ktensor(A));
B = W.*ktensor(A);

%% Compute normZ
if ~exist('normZsqr','var')
    normZsqr = norm(Z)^2;
end

% function value
f = 0.5 * normZsqr - innerprod(Z,B) + 0.5 * norm(B)^2;

% gradient computation
N = ndims(Z);
G = cell(N,1);
T = Z - B;
% G{1} = -T.data*A{2};
% G{2} = -T.data'*A{1};
 for n = 1:N
     G{n} = zeros(size(A{n}));
     G{n} = -mttkrp(T,A,n);
 end
 
 