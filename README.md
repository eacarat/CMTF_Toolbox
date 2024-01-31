# The MATLAB CMTF Toolbox 

The MATLAB CMTF Toolbox has two data fusion models based on Coupled Matrix and Tensor Factorizations to jointly analyze datasets in the form of matrices and higher-order tensors:
- CMTF (Coupled Matrix and Tensor Factorizations)
- Advanced CMTF (ACMTF)

In this toolbox, gradient-based all-at-once optimization algorithms are used to fit the models. The MATLAB CMTF Toolbox has the functions necessary to compute function values and gradients for CMTF 
and ACMTF. For the optimization routines, it uses the [Poblano Toolbox](https://github.com/sandialabs/poblano_toolbox) for unconstrained optimization, and the [LBFGSB implementation]( https://github.com/stephenbeckr/L-BFGS-B-C) 
for nonnegativity constraints. The [Tensor Toolbox](https://gitlab.com/tensors/tensor_toolbox) is also needed to run the functions in the Matlab CMTF Toolbox. 

You can get started with TEST_EXAMPLES illustrating example scripts, e.g., TESTER_CMTF, TESTER_ACMTF, TESTER_CMTF_MISSING, TESTER_ACMTF_MISSING showing the use 
of CMTF and ACMTF. For more details, explore CMTF_OPT and ACMTF_OPT. 

## What is new?
Compared to the previous versions of the MATLAB CMTF Toolbox [v1.1](http://www.models.life.ku.dk/joda/CMTF_Toolbox), the toolbox now has 
- Compatibility with the latest versions of Tensor Toolbox and Poblano Toolbox 
- Option to impose constraints through the use of lbfgsb
- Option to have a ridge penalty on the factor matrices when fitting a CMTF model 
- Real data example showing how to use the ACMTF model to jointly analyze measurements of mixtures from multiple platforms. That is the example used in the original ACMTF paper:
  E. Acar, E. E. Papalexakis, G. Gurdeniz,  M. A. Rasmussen,  A. J. Lawaetz, M. Nilsson,  R. Bro, [Structure-Revealing Data Fusion](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-15-239), BMC Bioinformatics, 15: 239, 2014.)


## How to cite
If you use the MATLAB CMTF Toolbox, please cite the toolbox along with the relevant publication:

- CMTF: E. Acar, T. G. Kolda, and D. M. Dunlavy, [All-at-once Optimization for Coupled Matrix and Tensor Factorizations](https://arxiv.org/abs/1105.3422), KDD Workshop on Mining and Learning with Graphs, 2011 

- ACMTF:
  - E. Acar, A. J. Lawaetz, M. A. Rasmussen, and R. Bro, [Structure-Revealing Data Fusion Model with Applications in Metabolomics](https://ieeexplore.ieee.org/document/6610925), Proceedings of 35th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (IEEE EMBC'13), pp. 6023-6026, 2013. 
  - E. Acar, E. E. Papalexakis, G. Gurdeniz,  M. A. Rasmussen,  A. J. Lawaetz, M. Nilsson,  R. Bro, [Structure-Revealing Data Fusion](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-15-239), BMC Bioinformatics, 15: 239, 2014.
