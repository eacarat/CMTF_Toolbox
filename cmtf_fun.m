function [f,g] = cmtf_fun(x, Z, Znormsqr)
% CMTF_FUN computes the function value and the gradient for coupled
% matrix-tensor factorization, where matrix model is a matrix factorization
% model and tensor model is a CANDECOMP/PARAFAC model.
% 
% [f,g] = cmtf_fun(x, Z, Znormsqr)
%
% Input:   x: a vector of length P, where for an I by J by K tensor and I
%             by L matrix, P = (I+J+K+L)*R and R is the number of
%             components. x = fac_to_vec(G), where G{1}:I by R, G{2}: J by
%             R, G{3}: K by R and G{4}: L by R. G{1}, G{2} and G{3} are the
%             factor matrices of the tensor factorization and G{1} and G{4} 
%             are the factor matrices of the matrix factorization.
%          Z: a structure with object, modes, size fields storing the
%             coupled data sets (See cmtf_check).
%          Znormsqr: a cell array with squared Frobenius norm of each Z.object
%
% Output:  f: function value of the combined objective function.
%          g: a vector of length P corresponding to the gradient.
%
% See also CMTF_OPT, CMTF_FG, CMTF_VEC_TO_FAC, TT_FAC_TO_VEC, PCA_FG, PCA_WFG, 
% TT_CP_FG, TT_CP_WFG, CMTF_CHECK. 
%
% This is the MATLAB CMTF Toolbox
% References: 
%    - (CMTF) E. Acar, T. G. Kolda, and D. M. Dunlavy, All-at-once Optimization for Coupled
%      Matrix and Tensor Factorizations, KDD Workshop on Mining and Learning
%      with Graphs, 2011 (arXiv:1105.3422v1)
%    - (ACMTF)E. Acar, A. J. Lawaetz, M. A. Rasmussen,and R. Bro, Structure-Revealing Data 
%      Fusion Model with Applications in Metabolomics, IEEE EMBC, pages 6023-6026, 2013.
%    - (ACMTF)E. Acar,  E. E. Papalexakis, G. Gurdeniz, M. Rasmussen, A. J. Lawaetz, M. Nilsson, and R. Bro, 
%      Structure-Revealing Data Fusion, BMC Bioinformatics, 15: 239, 2014.        
%

%% Convert the input vector into a cell array of factor matrices
A  = cmtf_vec_to_fac(x,Z);

%% Compute the function and gradient values
[f,G] = cmtf_fg(Z,A,Znormsqr);

% Vectorize the cell array of matrices
g = tt_fac_to_vec(G);

