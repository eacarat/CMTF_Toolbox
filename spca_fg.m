function [f,G] = spca_fg(Z, A, Znormsqr, beta)
% SPCA_FG Computes function and gradient of matrix factorization. 
% SPCA_FG also has the option of imposing sparsity on the weights of 
% rank-one matrices.
%
%   [f,G] = spca_fg(Z, A, Znormsqr, beta)
%
% Input: Z : a matrix 
%        A : a cell array of length three containing factor matrices and 
%            the weights of factors. 
%        Znormsqr: squared Frobenius norm of Z. 
%        beta: sparsity penalty parameter on the weights of rank-one matrices.
%
% Output: f: function value computed as
%                f = 0.5*||Z - A{1}*diag(A{3})*A{2}'||^2 
%                      + 0.5* beta * |A{3}|_1
%            where l1-penalty is replaced with a smooth approximation.
%         G: a cell array of length three with two matrices corresponding 
%            to the partial derivatives of the objective wrt to the factor
%            matrices as well as the last cell containing the partials for 
%            the weights of each factor. 
%
% See also SPCA_WFG, ACMTF_FG
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
Sigma = A{end};
A     = A(1:end-1);

U = A{1};
V = A{2};
UTU = U'*U;
VTV = V'*V;
SS  = Sigma*Sigma';

%% Function value
f1 = Znormsqr;
f2 = 0;
R = size(U,2);
for r = 1:R
  %  f2 = f2 + Sigma(r)*U(:,r)'*Z*V(:,r);
     f2 = f2 + Sigma(r)*ttv(Z,{U(:,r),V(:,r)},[1 2]);
end

W = UTU .* VTV.*SS;
f3 = sum(W(:));

f = 0.5 * f1 - f2 + 0.5 * f3;

% add sparsity constraint- l1 on Sigma
eps = 1e-8;
for r=1:R
    f = f + 0.5* beta*sqrt(Sigma(r)^2 + eps);    
end

%% Gradient

%factor matrices
ZV   = full(ttm(Z,V',2));
ZU   = full(ttm(Z,U',1));
G{1} = -khatrirao(ZV.data,Sigma') + U*(VTV.*SS);
G{2} = -khatrirao(ZU.data',Sigma') + V*(UTU.*SS);

% G{1} = -khatrirao(Z*V,Sigma') + U*(VTV.*SS);
% G{2} = -khatrirao(Z'*U,Sigma') + V*(UTU.*SS);

%sigma
G{3}=zeros(R,1);
for r=1:R
    %G{3}(r) = G{3}(r)-U(:,r)'*Z*V(:,r);
    G{3}(r) = G{3}(r)-ttv(Z,{U(:,r),V(:,r)},[1 2]);
end
G{3} = G{3} + sum(khatrirao(UTU.*VTV,Sigma'),2);
    
% sparsity constraint on sigma
for r=1:R
    G{3}(r) = G{3}(r) + 0.5*beta*Sigma(r)/sqrt(Sigma(r)^2+ eps);    
end


