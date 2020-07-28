function [f,G] = scp_wfg(Z, W, A, Znormsqr, beta)
% SCP_WFG Computes function and gradient of the CP function for data with
% missing entries. SCP_WFG also has the option of imposing sparsity on the 
% weights of rank-one components.
%
%   [f,G] = scp_wfg(Z, W, A, Znormsqr, beta) 
%
% Input: Z: an N-way tensor with missing entries replaced with 0.
%        W: an N-way indicator tensor containing zeros whenever 
%           data is missing (otherwise, ones).
%        A: a cell array of length N+1. The last cell corresponds to the 
%           weights of the components. 
%        Znormsqr: squared Frobenius norm of Z. 
%        beta: sparsity penalty parameter on the weights of rank-one tensors.
%
% Output: f: function value computed as 
%                f = 0.5*||W*(Z - ktensor(A{end}, A(1:end-1)))||^2 +
%                    + 0.5* beta *|A{end}|_1
%            where l1-penalty is replaced with a smooth approximation.
%         G: a cell array of length N+1, i.e., G{n}(:,r) is the partial 
%            derivative of the fit function with respect to A{n}(:,r), 
%            for n=1,..N. G{N+1}(r) is the partial derivative of the fit 
%            function wrt the norm of the rth factor. 
%  
% See also SCP_FG, ACMTF_FG
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
if ~isa(Z,'tensor') && ~isa(Z,'sptensor')
    error('Z must be a tensor or a sptensor');
end
N = ndims(Z);

if ~iscell(A) && ~isa(A,'ktensor');
    error('A must be a cell array or ktensor');
end

if isa(A,'ktensor')
    A = tocell(A);
end
R = size(A{1},2);

%%
Lambda = A{end};
A = A(1:end-1);
if isa(W,'sptensor')
    B = W.*ktensor(Lambda,A);
else
    B = W.*full(ktensor(Lambda,A));
end

%% function value
f_1 = Znormsqr;

% function value
f = 0.5 * f_1 - innerprod(Z,B) + 0.5 * norm(B)^2;

%% Calculate gradient 
G = cell(N+1,1);
T = Z-B;
for n = 1:N
    G{n} = -khatrirao(mttkrp(T,A,n),Lambda');    
end

% add sparsity constraint- l1 on lambda
eps = 1e-8;
for r=1:R
    f = f + 0.5* beta*sqrt(Lambda(r)^2 + eps);    
end

% Part of the gradient for Lambda
C = cell(R,N);
for n = 1:N
    for r = 1:R
        C{r,n} = A{n}(:,r);
    end
end
G{N+1} = zeros(R,1);
for r=1:R
     G{N+1}(r) = G{N+1}(r) -ttv(T, C(r,:),1:N);
end

% sparsity constraint on lambda
for r=1:R
    G{N+1}(r) = G{N+1}(r) + 0.5*beta*Lambda(r)/sqrt(Lambda(r)^2+ eps);    
end

