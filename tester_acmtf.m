function  data = tester_acmtf(data, varargin)
% TESTER_ACMTF Generates coupled data sets (in either dense or sptensor format)
% based on the optional input parameters and then uses ACMTF_OPT to fit a 
% coupled matrix-tensor factorization model with a goal of identifying 
% shared/unshared factors. The data sets in sparse format are generated using 
% factor matrices with sparse columns.
%
%   data = tester_acmtf; uses default settings to generate coupled data sets
%   and returns data, which is a structure with fields:
%           Zhat: factor matrices extracted using coupled matrix and tensor
%                 factorization, ktensor Zhat{p} has the weights of
%                 components and the factor matrices for the pth object.
%           Atrue: factor matrices used to generate the data sets
%           lambdas, lambdas_rec: weights used to generate the data sets and 
%                                 weights extracted using ACMTF, respectively.
%           out  : output of the optimization showing the stopping
%                  condition, gradient, etc.
%           Init : Initialization used for the optimization algorithm 
%
%   data = tester_acmtf('R',...) gives the number of components that will be
%   extracted from coupled data sets.
%
%   data = tester_acmtf('modes',...) gives as input how the modes are
%   coupled among each data set, e.g., {[1 2 3], [1 4], [2 5]}- generates
%   three data sets, where the third-order tensor shares the first mode with
%   the first matrix and the second mode with the second matrix.
%
%   data = tester_acmtf('size',...) gives as input the size of the data
%   sets, e.g., if we were to generate a third-order tensor of size 50 by 40
%   by 30 and a matrix of size 50 by 10 coupled in the first mode, then size
%   will be [50 40 30 10].
%
%   data = tester_acmtf('lambdas',...) gives as input the weights of each
%   component in each data set, e.g., {[1 1 0], [1 1 1]}, the first two
%   components are shared by both data sets while the last component is only
%   available in the second data. 
% 
%   data = tester_acmtf('init',...) gives as input the type of initialization 
%   to be used for the optimization algorithms, e.g.,'random', 'nvecs', or
%   a structure with fields fac and norms.
%
%   data = tester_acmtf('beta','...') gives as input the sparsity
%   penalty parameters.
%
%   data = tester_cmtf('flag_sparse',...): If flag_sparse is set to true for any of the
%   data sets, e.g., [1 0], sparse factor matrices are used to generate the coupled data
%   sets. Each data set will be stored in either dense or sparse format
%   depending on its flag_sparse value.
%
% See also ACMTF_OPT, TESTER_CMTF, CREATE_COUPLED, CREATE_COUPLED_SPARSE
%
% This is the MATLAB CMTF Toolbox, 2013.
% References: 
%    - (CMTF) E. Acar, T. G. Kolda, and D. M. Dunlavy, All-at-once Optimization for Coupled
%      Matrix and Tensor Factorizations, KDD Workshop on Mining and Learning
%      with Graphs, 2011 (arXiv:1105.3422v1)
%    - (ACMTF)E. Acar, A. J. Lawaetz, M. A. Rasmussen,and R. Bro, Structure-Revealing Data 
%      Fusion Model with Applications in Metabolomics, IEEE EMBC, pages 6023-6026, 2013.
%    - (ACMTF)E. Acar,  E. E. Papalexakis, G. Gurdeniz, M. Rasmussen, A. J. Lawaetz, M. Nilsson, and R. Bro, 
%      Structure-Revealing Data Fusion, BMC Bioinformatics, 15: 239, 2014.        
%
% Modifications: 
% September 2018: beta_cp and beta_pca are combined into a single input variable 
% where the sparsity penalty for each data set needs to be explicitly stated, and they all can be different.
% Feb 2020: Extending to generating nonnnegative factors and using lbfgsb as an option to fit ACMTF-OPT

%% Parse inputs
params = inputParser;
params.addParameter('R', 3, @(x) x > 0);
params.addParameter('alg', 'ncg', @(x) ismember(x,{'ncg','tn','lbfgs','lbfgsb'}));
params.addParameter('size', [50 30 40 20], @isnumeric);
params.addParameter('beta', [0 0], @isnumeric);
params.addParameter('modes', {[1 2 3], [1 4]}, @iscell);
params.addParameter('lambdas', {[1 1 1], [1 1 0]}, @iscell);
params.addParameter('flag_sparse',[0 0], @isnumeric);
params.addParameter('flag_gnn',[0 0 0 0], @isnumeric);
params.addParameter('flag_fnn',[0 0 0 0], @isnumeric);
params.addParameter('init', 'random', @(x) (isstruct(x) || ismember(x,{'random','nvecs'})));
params.parse(varargin{:});
%% Parameters
lambdas     = params.Results.lambdas;
modes       = params.Results.modes;
sz          = params.Results.size;
R           = params.Results.R;
init        = params.Results.init;
beta        = params.Results.beta;
flag_sparse = params.Results.flag_sparse;
flag_gnn     = params.Results.flag_gnn;     %how to generate data
flag_fnn     = params.Results.flag_fnn;     %how to impose nn
alg          = params.Results.alg;

%% Check parameters
if length(lambdas)~=length(modes)
    error('There should be weights for each data set');
end
P = length(modes);
for p=1:P
    l(p) = length(lambdas{p});
end
if length(unique(l))>1
    error('There should be the same number of weights for each data set');
end
if length(flag_sparse)<p
    error('flag_sparse should be specified for each data set');
end

%% Form coupled data
if isempty(data)
    if any(flag_sparse)
        % construct sparse data sets in the dense or sptensor format based on flag_sparse
        [X, Atrue] = create_coupled_sparse('size',sz,'modes',modes,'lambdas',lambdas,'flag_sparse',flag_sparse, 'flag_nn', flag_gnn);
    elseif ~any(flag_sparse) 
        % construct data sets in the dense format
        [X, Atrue] = create_coupled('size', sz, 'modes', modes, 'lambdas', lambdas,'flag_nn', flag_gnn);
    end
    P = length(X);
    for p=1:P
        Z.object{p} = X{p};
        Z.object{p} = Z.object{p}/norm(Z.object{p});    
    end
    Z.modes = modes;
    Z.size  = sz;
else
    Z     = data.Z;
    Atrue = data.Atrue;
end

%% Fit ACMTF using one of the first-order optimization algorithms 
options = ncg('defaults');
options.Display ='final';
options.MaxFuncEvals = 100000;
options.MaxIters     = 10000;
options.StopTol      = 1e-8;
options.RelFuncTol   = 1e-8;

% fit ACMTF-OPT
[Zhat,G,out]    = acmtf_opt(Z,R,'init',init,'alg_options',options, 'beta', beta, 'flag_nn', flag_fnn, 'alg',alg);        
data.Zhat       = Zhat;
data.out        = out;
data.Atrue      = Atrue;
l_rec = zeros(P, R);
for p =1:P
    temp        = normalize(Zhat{p});
    l_rec(p,:)  = temp.lambda;    
end
tt=[];
for i=1:length(lambdas)
    tt = [tt; lambdas{i}];
end
data.lambdas    = tt;
data.lambda_rec = l_rec;
data.Init       = G;
data.Z          = Z;
