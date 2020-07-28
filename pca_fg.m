function [f,G] = pca_fg(Z,A,Znormsqr)
% PCA_FG Function value and gradient of matrix factorization.
%
% [f,G] = pca_fg(Z,A,Znormsqr)
%
% Input:  Z: data matrix to be factorized using matrix factorization
%         A: a cell array of two factor matrices
%         Znormsqr: squared Frobenius norm of Z
%
% Output: f: function value, i.e., f = (1/2) ||Z - A{1}*A{2}'||^2
%         G: a cell array of two matrices corresponding to the gradients
%
% See also CMTF_FG, PCA_WFG
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

%%
% if isa(Z, 'tensor')
%     Z=Z.data;
% end    
    
%A = normalize(ktensor(A),0);
U = A{1};
V = A{2};
UTU = U'*U;
VTV = V'*V;

if exist('Znormsqr','var')
    f1 = Znormsqr;
else
    f1 = sum(Z(:).^2);
end

f2 = 0;
R = size(U,2);
for r = 1:R
    %f2 = f2 + U(:,r)'*Z*V(:,r);
    f2 = f2 + ttv(Z,{U(:,r),V(:,r)},[1 2]);
end

W = UTU .* VTV;
f3 = sum(W(:));

f = 0.5 * f1 - f2 + 0.5 * f3;

% G{1} = -Z*V + U*VTV;
% G{2} = -Z'*U + V*UTU;

ZV   = full(ttm(Z,V',2));
ZU   = full(ttm(Z,U',1));
G{1} = -ZV.data + U*VTV;
G{2} = -ZU.data' + V*UTU;
