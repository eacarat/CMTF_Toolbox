function [f,G] = scp_fg(Z, A, Znormsqr, beta)
% SCP_FG Computes function and gradient of the CP function. SCP_FG also 
% has the option of imposing sparsity on the weights of rank-one components.
%
%   [f,G] = scp_fg(Z, A, Znormsqr, beta) 
%           
% Input:  Z: an N-way tensor 
%         A: a cell array of length N+1. The last cell corresponds to 
%           the weights of the components. 
%         Znormsqr: norm of Z. 
%         beta: sparsity penalty parameter on the weights of rank-one tensors.
%
% Output: f: function value computed as 
%               f = 0.5*||Z - ktensor(A{end}, A(1:end-1))||^2  
%                   + 0.5*beta*|A{end}|_1 
%            where l1-penalty is replaced with a smooth approximation.
%         G: a cell array of length N+1, i.e., G{n}(:,r) is the partial 
%            derivative of the fit function with respect to A{n}(:,r), 
%            for n=1,..N. G{N+1}(r) is the partial derivative of the fit 
%            function wrt the norm of the rth factor. 
%
% See also SCP_WFG, ACMTF_FG
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

%% Upsilon and Gamma
Upsilon = cell(N,1);
for n = 1:N
    Upsilon{n} = A{n}'*A{n};
end

Gamma = cell(N,1);
for n = 1:N
    Gamma{n} = ones(R,R);
    for m = [1:n-1,n+1:N]
        Gamma{n} = Gamma{n} .* Upsilon{m};
    end
end

LL   =  Lambda*Lambda';
Teta =  Gamma{1} .* Upsilon{1};    

%% f1
f_1 = Znormsqr;

%% Calculate gradient 
G = cell(N+1,1);
U = khatrirao(mttkrp(Z,A,1), Lambda');
V = A{1} .* U;
f_2 = sum(V(:));
G{1} = -U + A{1}*(LL.*Gamma{1});
for n = 2:N
    U = khatrirao(mttkrp(Z,A,n),Lambda');
    G{n} = -U + A{n}*(LL.*Gamma{n});
end

%F3
%W   = Gamma{1} .* Upsilon{1}.*LL;
W   = Teta.*LL;
f_3 = sum(W(:));

%SUM of pieces of the loss function
f = 0.5 * f_1 - f_2 + 0.5 * f_3;

% add sparsity constraint- l1 on lambda
eps = 1e-8;
for r=1:R
    f = f + 0.5* beta*sqrt(Lambda(r)^2 + eps);    
end

% Part of the gradient for Lambda
C =cell(R,N);
for n=1:N
    for r=1:R
        C{r,n} = A{n}(:,r);
    end
end
G{N+1} = zeros(R,1);
for r=1:R
     G{N+1}(r) = G{N+1}(r) -ttv(Z, C(r,:),1:N);
end
G{N+1} = G{N+1} + sum(khatrirao(Teta,Lambda'),2);

% sparsity constraint on lambda
for r=1:R
    G{N+1}(r) = G{N+1}(r) + 0.5*beta*Lambda(r)/sqrt(Lambda(r)^2+ eps);    
end

