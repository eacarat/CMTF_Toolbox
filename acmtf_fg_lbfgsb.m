function [f,G] = acmtf_fg_lbfgsb(Z, A, Znormsqr,beta)
% ACMTF_FG_LBFGSB Function and gradient of coupled matrix and tensor factorization, 
% where the coupled model is formulated, for instance, for a third-order tensor 
% and a matrix coupled in the first mode as 
% f = 0.5*||Z.object{1}-[|\Lambda; A, B, C|]||^2 
%     + 0.5* ||Z.object{2} -A\SigmaD'||^2
%     + 0.5*beta(1)*sum(\Lambda) + 0.5*beta(2) *sum(diag(\Sigma)) + P, 
% where P indicates quadratic penalty terms to normalize the columns of 
% factor matrices to unit norm, and \Lambda and diag(\Sigma) are constrained to be nonnegative.
% In ACMTF_FG, there are no nonnegativity constraints on the weights
% (\Lambda, \Sigma) while, here, weights are constrained to be nonnegative;
% therefore, the formulation of the loss is slightly different.
%
% [f,G] = acmtf_fg_lbfgsb(Z, A, Znormsqr, beta) 
%
% Input:  Z: a struct with object, modes, size, miss fields storing the 
%            coupled data sets (See cmtf_check)
%         A: a struct with fac (a cell array corresponding to factor 
%            matrices) and norms (a cell array corresponding to the weights
%            of components in each data set) fields. 
%         Znormsqr: a cell array with squared Frobenius norm of each Z.object
%         beta : array of sparsity parameters on the weights of rank-one components in each data set
% 
% Output: f: function value
%         G: a struct with fac (i.e., G.fac corresponding to the part of the 
%            gradient for the factor matrices) and norms (i.e., G.norms 
%            corresponding to the part of the gradient for the weights of
%            the components in each data set) fields.
%
% See also ACMTF_FUN_LBFGSB, SPCA_FG_LBFGSB, SPCA_WFG_LBFGSB, SCP_FG_LBFGSB, SCP_WFG_LBFGSB
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

if ~isstruct(A)
    error('A must be a structure with fields: fac and norms, each being a cell array');
end

P = numel(Z.object);

fp   = cell(P,1);
Gp   = cell(P,1);
for p = 1:P   
    B = [A.fac(Z.modes{p}); A.norms{p}];
    if length(size(Z.object{p}))>=3
        % Tensor
        if isfield(Z,'miss') && ~isempty(Z.miss{p})            
            [fp{p},Gp{p}] = scp_wfg_lbfgsb(Z.object{p}, Z.miss{p}, B, Znormsqr{p}, beta(p));        
        else
            [fp{p},Gp{p}] = scp_fg_lbfgsb(Z.object{p}, B, Znormsqr{p}, beta(p));        
        end
    elseif length(size(Z.object{p}))==2
        % Matrix
        if isfield(Z,'miss') && ~isempty(Z.miss{p})
            [fp{p},Gp{p}] = spca_wfg_lbfgsb(Z.object{p}, Z.miss{p}, B, Znormsqr{p}, beta(p));                        
        else
            [fp{p},Gp{p}] = spca_fg_lbfgsb(Z.object{p}, B, Znormsqr{p}, beta(p));                         
        end
    end
    G.norms(p) = Gp{p}(end);    
end

%% Compute overall gradient
for n = 1:numel(Z.size)
    G.fac{n} = zeros(size(A.fac{n}));
end
for p = 1:P
    for i = 1:length(Z.modes{p})
        j = Z.modes{p}(i);
        G.fac{j} = G.fac{j} + Gp{p}{i};
    end    
end

%% Compute overall function value
f = sum(cell2mat(fp));

