function  data = tester_cmtf(data, varargin)
% TESTER_CMTF Generates coupled data sets (in either dense or sptensor format)
% based on the optional input parameters and then uses CMTF_OPT to fit a 
% coupled matrix-tensor factorization model. The data sets in sparse format
% are generated using factor matrices with sparse columns.
%
%   data = tester_cmtf; uses default settings to generate coupled data sets
%   and returns data, which is a structure with fields:
%           Fac    : a cell array of factor matrices extracted using CMTF
%           Factrue: a cell array of factor matrices used to generate data
%           out    : output of the optimization showing the stopping
%                    condition, gradient, function value, etc.
%           Init   : Initialization used for the optimization algorithm 
%
%   data = tester_cmtf('R',...) gives the number of components that will be
%   extracted from coupled data sets.
%
%   data = tester_cmtf('modes',...) gives as input how the modes are
%   coupled among each data set, e.g., {[1 2 3], [1 4], [2 5]}- generates
%   three data sets, where the third-order tensor shares the first mode with
%   the first matrix and the second mode with the second matrix.
%
%   data = tester_cmtf('size',...) gives as input the size of the data
%   sets, e.g., if we were to generate a third-order tensor of size 50 by 40
%   by 30 and a matrix of size 50 by 10 coupled in the first mode, then size
%   will be [50 40 30 10].
%
%   data = tester_cmtf('lambdas',...) gives as input the weights of each
%   component in each data set, e.g., {[1 1 1], [1 1 1]}. The length of each
%   cell should be the same.
% 
%   data = tester_cmtf('init',...) gives as input the type of initialization
%   to be used for the optimization algorithms, e.g.,'random', 'nvecs', or a 
%   cell array of matrices.
%
%   data = tester_cmtf('flag_sparse',...): If flag_sparse is set to true for any of the
%   data sets, e.g., [1 0], sparse factor matrices are used to generate the coupled data
%   sets. Each data set will be stored in either dense or sparse format
%   depending on its flag_sparse value.
%
% See also CMTF_OPT, TESTER_ACMTF, CREATE_COUPLED, CREATE_COUPLED_SPARSE
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
% Jan 2020: Added lbfgsb option to CMTF-OPT

%% Parse inputs
params = inputParser;
params.addParameter('R', 3, @(x) x > 0);
params.addParameter('alg', 'ncg', @(x) ismember(x,{'ncg','tn','lbfgs','lbfgsb'}));
params.addParameter('size', [50 30 40 20], @isnumeric);
params.addParameter('modes', {[1 2 3], [1 4]}, @iscell);
params.addParameter('lambdas', {[1 1 1], [1 1 1]}, @iscell);
params.addParameter('flag_gnn',[0 0 0 0], @isnumeric); %nonnegative data generation
params.addParameter('flag_fnn',[0 0 0 0], @isnumeric); %nonnegative model fitting
params.addParameter('flag_sparse',[0 0],@isnumeric); %meaning sparse storage here!
params.addParameter('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
params.parse(varargin{:});

%% Parameters
lambdas     = params.Results.lambdas;
modes       = params.Results.modes;
sz          = params.Results.size;
R           = params.Results.R;
init        = params.Results.init;
flag_sparse = params.Results.flag_sparse;
flag_gnn     = params.Results.flag_gnn;     %how to generate data
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


%% Form coupled data
if isempty(data)
    if any(flag_sparse)
        % construct sparse data sets in the dense or sptensor format based on flag_sparse
        [X, Atrue] = create_coupled_sparse('size',sz,'modes',modes,'lambdas',lambdas,'flag_sparse',flag_sparse, 'flag_nn', flag_gnn);
    elseif ~any(flag_sparse) 
        % construct data sets in the dense format
        [X, Atrue] = create_coupled('size', sz, 'modes', modes, 'lambdas', lambdas, 'flag_nn', flag_gnn);
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
    Atrue = data.Factrue;
end


%% Fit CMTF using one of the first-order optimization algorithms 
options = ncg('defaults');
options.Display ='final';
options.MaxFuncEvals = 100000;
options.MaxIters     = 10000;
options.StopTol      = 1e-8;
options.RelFuncTol   = 1e-8;

% fit CMTF-OPT
[Fac,G,out]   = cmtf_opt(Z,R,'init',init,'alg_options',options,'flag_nn', params.Results.flag_fnn, 'alg',alg); 
data.Fac      = Fac.U;
data.out      = out.OptOut;
data.Factrue  = Atrue;
data.Init     = G; 
data.Z = Z;
