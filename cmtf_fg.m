function [f,G] = cmtf_fg(Z,A,Znormsqr,lambda)
% CMTF_FG Function and gradient of coupled matrix-tensor factorization,
% where the coupled model is formulated, for instance, for a third-order tensor 
% and a matrix coupled in the first mode as 
% f = 0.5*||Z.object{1}-[|A, B, C|]||^2 + 0.5* ||Z.object{2} -AD'||^2
%
% [f,G] = cmtf_fg(Z,A,Znormsqr)
%
% Input:  Z: a struct with object, modes, size, miss fields storing the 
%            coupled data sets (See cmtf_check)
%         A: a cell array of factor matrices
%         Znormsqr: a cell array with squared Frobenius norm of each Z.object
%         lambda: ridge penalty parameter
%
% Output: f: function value
%         G: a cell array of gradients corresponding to each factor matrix;
%            in other words, G{n}(:,r) is the partial derivative of the 
%            objective function with respect to A{n}(:,r).  
%
% See also CMTF_FUN, PCA_FG, PCA_WFG, TT_CP_FG, TT_CP_WFG, CMTF_CHECK
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


if isa(A,'ktensor')
    A = tocell(A);
end

if ~iscell(A)
    error('A must be a cell array');
end

P = numel(Z.object);

if ~exist('Znormsqr','var')
    Znormsqr = cell(P,1);
    for p = 1:P
        if isa(Z.object{p},'tensor') || isa(Z.object{p},'sptensor')
            Znormsqr{p} = norm(Z.object{p})^2;
        else
            Znormsqr{p} = norm(Z.object{p},'fro')^2;
        end
    end
end

fp = cell(P,1);
Gp = cell(P,1);
for p = 1:P   
    if length(size(Z.object{p}))>=3
        % Tensor
        if isfield(Z,'miss') && ~isempty(Z.miss{p})            
            [fp{p},Gp{p}] = tt_cp_wfg(Z.object{p}, Z.miss{p}, A(Z.modes{p}), Znormsqr{p});                
        else            
            [fp{p},Gp{p}] = tt_cp_fg(Z.object{p}, A(Z.modes{p}), Znormsqr{p});
        end       
    elseif length(size(Z.object{p}))==2
        % Matrix
        if isfield(Z,'miss') && ~isempty(Z.miss{p})
            [fp{p},Gp{p}] = pca_wfg(Z.object{p}, Z.miss{p}, A(Z.modes{p}), Znormsqr{p});
        else        
            [fp{p},Gp{p}] = pca_fg(Z.object{p}, A(Z.modes{p}), Znormsqr{p});        
        end
    end
end

%% Compute overall gradient
G = cell(size(A));
for n = 1:numel(G)
    %G{n} = zeros(size(A{n}));
    G{n} = lambda*A{n};
end
for p = 1:P
    for i = 1:length(Z.modes{p})
        j = Z.modes{p}(i);
        G{j} = G{j} + Gp{p}{i};
    end
end

%% Compute overall function value
f = sum(cell2mat(fp));
freg = 0;
for n=1:numel(G)
    freg =  freg + 0.5*lambda*norm(A{n},'fro')^2;
end
f = f+freg;
return;
