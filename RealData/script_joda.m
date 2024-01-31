% This script jointly analyzes the DOSY NMR and LCMS data using ACMTF as
% described in 
%      E. Acar,  E. E. Papalexakis, G. Gurdeniz, M. Rasmussen, A. J. Lawaetz, M. Nilsson, and R. Bro, 
%      Structure-Revealing Data Fusion, BMC Bioinformatics, 15: 239, 2014.      
%
% More details on the data sets are available https://ucphchemometrics.com/joda/
%
% All three data sets (EEM, NMR and LCMS) are jointly analyzed in 
%    C. Schenker, J. E. Cohen, E. Acar, A Flexible Optimization Framework for Regularized Matrix-Tensor 
%    Factorizations with Linear Couplings, IEEE Journal of Selected Topics in Signal Processing, 15(3), 2021
%
%% Load coupled 3-way DOSY NMR and 2-way LC-MS
load('EEM_NMR_LCMS.mat')
%% Prepare the coupled object to be given as an input to ACMTF_OPT
clear X
modes = {[1 2 3], [1 4]};
YY    = Y(:,1:10:end,:); % downsample NMR in the chemical shifts mode
X{1}  = YY;
AA    = Z; clear Z
X{2}  = AA;
P     = length(X);
for i = 1:P
    Z.object{i} = X{i}.data;
    Z.modes{i}  = modes{i};
end
Z.size  = [size(X{1}) size(X{2},2)];

%% Missing data: 
% Check if there are missing entries. If so, record the place of missing entries in Z.miss{i} for the ith data set 
for i=1:length(Z.object)
    if ~isempty(find(isnan(Z.object{i})==1))
        Zmiss       = Z.object{i};    
        W           = ones(size(Zmiss));
        Worg        = find(isnan(Zmiss)==1); 
        Zmiss(Worg) = 0;
        W(Worg)     = 0;
        W           = tensor(W);
        Z.miss{i}   = W;
        Z.object{i} = tensor(Zmiss);
    else
        Z.object{i} = tensor(Z.object{i});
        Z.miss{i}   =[];
    end    
end

%% Preprocessing: 
% scale the LC-MS peaks by the standard deviation
Z.object{2} = tensor(Z.object{2}.data*diag(1./std(Z.object{2}.data)));
% normalize data sets by their Frobenius norm
norms = zeros(P,1);
for i = 1:P
    norms(i)    = norm(Z.object{i});
    Z.object{i} = Z.object{i}/norms(i);
end

%% Fit ACMTF
% Set parameters of the optimization algorithm
options = ncg('defaults');
options.Display ='iter';
options.DisplayIters  = 100;
options.MaxFuncEvals = 100000;
options.MaxIters     = 10000;
options.StopTol      = 1e-10;
options.RelFuncTol   = 1e-8;

% Set the model parameters
R =6;              % number of components 
beta      = [0.001 0.001]; % l1-penalty parameter for the higher-order tensors
nb_starts = 5;    % number of different initializations

for i=1:nb_starts
    if i==1
        [Fac{i},~,out{i}] = acmtf_opt(Z,R,'init','nvecs','alg_options',options, 'beta', beta, 'alg', 'ncg');        
    else
        [Fac{i},~,out{i}] = acmtf_opt(Z,R,'init','random','alg_options',options, 'beta', beta, 'alg', 'ncg');        
    end
end

%% Check whether the run returning the smallest function value has converged and record the output from that run
for i=1:nb_starts
    f(i) = out{i}.F;
end

[ff, index] = sort(f,'ascend');
%check the gradient
g2normx = norm(out{index(1)}.G)/length(out{index(1)}.G);
if (g2normx < 1e-5) && (out{index(1)}.ExitFlag~=2) && (out{index(1)}.ExitFlag~=1)
    Zhat = Fac{index(1)};
else 
    Zhat = {};
end
Out  = out{index(1)};

l_rec = zeros(P, R);
for p = 1:P
    temp        = normalize(Zhat{p});
    l_rec(p,:)  = temp.lambda;    
end
data.Zhat       = Zhat;
data.out        = Out;
data.lambda_rec = l_rec;
data.X          = X;
data.Z          = Z;
data.fits       = ff;
%% Compare with the true design of the experiment
load('TrueDesign.mat')
TrueDesign=A([1:26 28:end],:);
corr(data.Zhat{1}.U{1}, TrueDesign)


