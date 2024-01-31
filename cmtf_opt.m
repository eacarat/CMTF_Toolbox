function [P, G, output] = cmtf_opt(Z,R,varargin)
% CMTF_OPT Fits a coupled matrix and tensor factorization (CMTF) model using 
% first-order optimization.
%
%   A = CMTF_OPT(Z,R) fits an R-component CMTF model to the coupled 
%   data stored in Z and returns the factor matrices for each in A.
%   Z is a structure with object, modes, size, miss fields storing 
%   the coupled data sets (See cmtf_check)
%
%   A = CMTF_OPT(Z,R,'param',value,...) specifies additional
%   parameters for the method. Specifically...
%
%   'alg' - Specifies optimization algorithm {'ncg'}
%      'ncg'   Nonlinear Conjugate Gradient Method
%      'lbfgs' Limited-Memory BFGS Method
%      'tn'    Truncated Newton
%      'lbfgsb' LBFGS-B with bound constraints (choice of this algorithm,
%      imposes nonnegativity constraints on the weights by default, and then
%      additional constraints on the factor matrices can be imposed using
%      flag_nn)
%
%   'init' - Initialization for component matrices {'random'}. This
%   can be a cell array with the initial matrices or one of the
%   following strings:
%      'random' Randomly generated via randn function
%      'nvecs'  Selected as leading left singular vectors of unfolded data 
%               in Z.object{n}
%
%   'flag_nn' - Binary array of length equal to the number of factor
%   matrices indicating whether the factors in each mode are nonnegative or
%   not, e.g., flag_nn=[1 1 0 0] is used to impose nonnegativity constraints 
%   in the first two modes.
%
%   'alg_options' - Parameter settings for selected optimization
%   algorithm. For example, type OPTIONS = NCG('defaults') to get
%   the NCG algorithm options which can then be modified and passed
%   through this function to NCG.
%
%   'ridge_penalty' - Parameter for the ridge penalty on the factor matrices -
%    the default value is 0.
%
%   [A, G0] = CMTF_OPT(...) also returns the initial guess.
%
%   [A, G0, OUT] = CMTF_OPT(...) also returns a structure with the
%   optimization exit flag, the final relative fit, and the full
%   output from the optimization method.
%
% See also CMTF_FUN, CMTF_FG, CMTF_VEC_TO_FAC, TT_FAC_TO_VEC,
% PCA_FG, PCA_WFG, TT_CP_FG, TT_CP_WFG, CMTF_CHECK, CMTF_NVECS.
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

%% Error checking
cmtf_check(Z);

if (nargin < 2)
    error('Error: invalid input arguments');
end

%% Set parameters
params = inputParser;
params.addParameter('alg', 'ncg', @(x) ismember(x,{'ncg','tn','lbfgs', 'lbfgsb'}));
params.addParameter('ridge_penalty', 0,  @isnumeric); 
params.addParameter('flag_nn', [0 0 0 0],  @isnumeric);
params.addParameter('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
params.addOptional('alg_options', '', @isstruct);
params.parse(varargin{:});
P = numel(Z.object);

use_lbfgsb = strcmp(params.Results.alg,'lbfgsb');
lambda     = params.Results.ridge_penalty;

%% Set up optimization algorithm
if use_lbfgsb % L-BFGS-B
    if ~exist('lbfgsb','file')
        error(['CMTF_OPT requires L-BFGS-B function. This can be downloaded'...
            'at https://github.com/stephenbeckr/L-BFGS-B-C']);
    end
else % POBLANO
    switch (params.Results.alg)
        case 'ncg'
            fhandle = @ncg;
        case 'tn'
            fhandle = @tn;
        case 'lbfgs'
            fhandle = @lbfgs;
    end
     if ~exist('poblano_params','file')
        error(['CMTF_OPT requires Poblano Toolbox for Matlab. This can be ' ...
            'downloaded at http://software.sandia.gov/trac/poblano.']);
    end  
end

%% Set up optimization algorithm options
if use_lbfgsb
    options.maxIts      = 10000;
    options.maxTotalIts = 50000;
    options.printEvery  = 100;
   % options.factr = 1e-9/eps;
elseif isempty(params.Results.alg_options)
    options = feval(fhandle, 'defaults');
else
    options = params.Results.alg_options;
end
     

%% Initialization
sz = Z.size;
N = length(sz);

if iscell(params.Results.init)
    G = params.Results.init;
elseif strcmpi(params.Results.init,'random')
    G = cell(N,1);
    for n=1:N
        G{n} = randn(sz(n),R);
        if use_lbfgsb
            if params.Results.flag_nn(n)==1
                G{n} = rand(sz(n),R);
            end
        end
        for j=1:R
            G{n}(:,j) = G{n}(:,j) / norm(G{n}(:,j));
        end
    end
elseif strcmpi(params.Results.init,'nvecs')
    G = cell(N,1);
    for n=1:N
        G{n} = cmtf_nvecs(Z,n,R);
    end
else
    error('Initialization type not supported')
end

if use_lbfgsb
    NN = sum(sz)*R; 
    l  = -inf(NN,1);    
    for n = 1:length(sz)
       if params.Results.flag_nn(n)==1
           l(sum(sz(1:n-1))*R+1:sum(sz(1:n))*R) = zeros(sz(n)*R,1);
       end
    end
    u  = inf(NN,1);      
end

%% Fit CMTF using Optimization
Znormsqr = cell(P,1);
for p = 1:P
    if isa(Z.object{p},'tensor') || isa(Z.object{p},'sptensor')
        Znormsqr{p} = norm(Z.object{p})^2;
    else
        Znormsqr{p} = norm(Z.object{p},'fro')^2;
    end
end
if  use_lbfgsb
    options.x0 = tt_fac_to_vec(G);
    fhandle    = @(x)cmtf_fun(x,Z,Znormsqr, lambda);
    [out.X,out.F, out.info_lbfgsb] = lbfgsb(fhandle, l, u, options);
else
    out = feval(fhandle, @(x)cmtf_fun(x,Z,Znormsqr, lambda), tt_fac_to_vec(G), options);
end


%% Compute factors and model fit
P = ktensor(cmtf_vec_to_fac(out.X, Z));
if nargout > 2
    if use_lbfgsb
        output.info_lbfgsb = out.info_lbfgsb;
        output.X = out.X;
        output.F = out.F;
    else
        output.ExitFlag  = out.ExitFlag;
    end
    output.OptOut = out;
end

