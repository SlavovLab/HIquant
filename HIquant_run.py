import collections
import numpy as np
import numpy.matlib as mtl
import pandas as pd
import os
import sys
import scipy.sparse as sps
from scipy.sparse.csgraph import connected_components
import numba
import time
from itertools import combinations
import HIquant_functions as hf
import Solvers
from cvxopt import solvers, matrix, lapack
import matplotlib.pyplot as plt
import HIquant_test_path_set as ts
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
os.getcwd()

# Ed - adapted script for command line definition of input and output files
# Note: script requires 1 input and 2 output files
# Input: should be in the format: protein IDs, peptide sequence (or unique identifier), conditions 1:n
# Output 1: Proteoform abundance in each condition
# Output 2: Error and confidence measures for the proteoform inference

print(f"HIquant input: {sys.argv[1]=}")
print(f"HIquant output: {sys.argv[2]=}")
print(f"HIquant error output: {sys.argv[3]=}")
# Ed CMD line modification end

print( 'Importing data from the directory below ...')
st=time.time()

#test_path = '/Users/ed/Documents/GitHub/HIquant/edtestPAICS.txt'
test_path = sys.argv[1]

Mat_A , Mat_S, Mat_CC , Mat_nC, Dat_data, Dat_textdata, Dat_proteins, Data_isunique, Num_peptides, Num_proteins = hf.GetData(test_path)
ed=time.time()
print('Time:',ed-st)

Mat_nC
Num_proteins
Num_peptides
Dat_data

## 2. Complie a list protein having unique peptde and common peptides.
proteins_uniq_pep = Dat_textdata[Data_isunique==1].iloc[:,0].tolist()
# proteins_uniq_indx = pd.DataFrame(Dat_proteins).isin(proteins_uniq_pep)
# print('The number of proteins with unique peptides is :', proteins_uniq_indx.sum().values) % redundont, use the following
print('The number of proteins with unique peptides is :', len(np.unique(proteins_uniq_pep)))
print('Number of proteins ONLY with common peptides:', Num_proteins-len(np.unique(proteins_uniq_pep)))


## 3. Main loop over the connected_components
# some pre-difined variable
cluster_protein_name = dict()
# The feature variables initialization
Xi = np.array([])
X_corr_mean = np.array([])
CV_mean = np.array([])
CV_max = np.array([])
CV_min = np.array([])
P_eigen_sp = np.array([])
Neg_fract = np.array([])
C = 0
non_empty_ind = np.zeros((Mat_nC,1))
# Inferred variables
P_QP_rsq = np.array([])
P_homo_CD = dict()
P_homo_QP = dict()
P_homo_SVD = dict()

CD_homo_P = dict()
CD_prot_sub = dict()
CD_opt = dict()

SVD_homo_P = dict()

QP_homo_P = dict()
QP_prot_sub = dict()
QP_opt = dict()

# -----------*************************************************------------------------
# -----------loop over all connected components to solve each homologous protein group
# -----------*************************************************------------------------
st=time.time()

cc_i =0
for cc_i in range(Mat_nC):
    #For each solver we need to have prot_sub for each CC
    # 1). ------------------------return protein index associated with particular homologs group
    cc_i_homo, = np.where(np.in1d(Mat_CC,cc_i))
#     print('The CC {} has protein index {}'.format(cc_i,cc_i_homo))
    if len(cc_i_homo) == 1:
        continue

    # 2). ------------------------Compile a stoichemetric matrix S and data for cc_i_Homo (a connected component)
    inds_peptides_i, = np.where(Mat_S[:, cc_i_homo].sum(axis=1)>0)
#     print('peptide indices:',inds_peptides_i)
    prot_S = Mat_S[np.ix_(inds_peptides_i, cc_i_homo)]
#     print(prot_S)
    prot_dat = Dat_data.iloc[inds_peptides_i,:]
#     print('S matrix is: {}'.format(prot_S))
#     print('Data matrix is: {}'.format(prot_dat))


    # 3). ------------------------Get the protein names of cluster cc_i and store in a dictionary
    cluster_protein_name[str(cc_i)] = Dat_proteins[cc_i_homo]

    # 4). ------------------------Colapses data
    prot_sub_S, prot_sub_dat = hf.mSX2( prot_S, prot_dat.values)
    prot_sub = dict()
    prot_sub['S'] = prot_sub_S
    prot_sub['X'] = prot_sub_dat
    null_dim = hf.null_sp_dim(prot_sub)
    null_dim
    if null_dim > 2:
        continue
    # 5). ------------------------Calculate features for each solver
    # 5).1 ------------------------The common feature for all solvers
    ## ----- The ||X_i|| for each protein

    # print(prot_sub_dat)
    # print(np.std(prot_sub_dat,axis=1))
    # print(np.mean(prot_sub_dat,axis=1))
    P_relative_norm_temp = np.std(prot_sub['X'],axis=1) / np.mean(prot_sub['X'],axis=1)
    P_relative_norm = np.nanmean(P_relative_norm_temp)
    Xi = np.append(Xi, P_relative_norm)

    ## ----- The mean correlation between the rows of X as a measure of linear independence between the the columns of X.
    ##  Note: that in matlab when row ==2, the corr matrix always =1
    X_corr = np.corrcoef(prot_sub['X'].T)
    X_corr_modi = X_corr - np.eye(X_corr.shape[0]) # the diagnal of X_corr shows 1, but when subtract by np.eye, the result shows a close to zeros Number
    X_corr_mean_i = hf.mean_squareform(X_corr_modi)
    X_corr_mean = np.append(X_corr_mean, X_corr_mean_i)

    # 5.2 ------------------------The features specific to different solvers
    ## ----- -\-\-\-\-\-\-\----♌ The CD solver inference ----/-/-/-/-/-/------------
    CD_solution = Solvers.CD(prot_sub, 1000)
    CD_protein, opt_CD = CD_solution
    CD_homo_P[str(cc_i)] = CD_protein
    CD_prot_sub[str(cc_i)] = prot_sub
    CD_opt[str(cc_i)] = opt_CD


    ## ----- -\-\-\-\-\-\-\----♌ The QP solver inference ----/-/-/-/-/-/------------
    QP_solution = Solvers.QP(prot_sub)
    QP_protein, opt_QP = QP_solution
    QP_homo_P[str(cc_i)] = QP_protein
    QP_prot_sub[str(cc_i)] = prot_sub
    QP_opt[str(cc_i)] = opt_QP
    P_QP_rsq = np.append(P_QP_rsq, opt_QP['rsq'])
    # 3 CV score features associated with QP solver
    cv_mean_i = hf.cv_score(QP_protein, 'mean')
    cv_max_i = hf.cv_score(QP_protein, 'max')
    cv_min_i = hf.cv_score(QP_protein, 'min')
    CV_mean = np.append(CV_mean, cv_mean_i)
    CV_max = np.append(CV_max, cv_max_i)
    CV_min = np.append(CV_min, cv_min_i)



    ## ----- -\-\-\-\-\-\-\----♌ The SVD solver inference ----/-/-/-/-/-/------------
    SVD_solution = Solvers.SVD(prot_sub)
    SVD_protein, opt_SVD = SVD_solution
    SVD_homo_P[str(cc_i)] = SVD_protein
    P_eigen_sp = np.append(P_eigen_sp, opt_SVD['eigen_spacing']) # 𝕴-> Common features: eigen spacing

    vi = opt_SVD['V']
    vi = np.median(np.sign(vi))*vi
    Neg_fract_i = np.sum(vi<0)/len(vi)
    Neg_fract = np.append(Neg_fract, Neg_fract_i)
    # print('The protein homologs group {} has {} completed'.format(cc_i,len(cc_i_homo)))

    C = C + 1
    non_empty_ind[cc_i] = 1

print('finished looping over connected_components')
ed=time.time()
print('Time for looping:',ed-st)




# Obatain non_empty (a homo_group has >2 proteins) # in general, I dont need mask
# mask = np.concatenate(non_empty_ind >0)[0:stop+1]
# ♌ Getting the P inferred stacked vector matrix
# P_CD = np.array(list([np.concatenate(CD_homo_P[str(cc_i)],axis=0) for cc_i in range(stop+1)]))[mask]
# P_SVD = np.array(list([np.concatenate(SVD_homo_P[str(cc_i)],axis=0) for cc_i in range(stop+1)]))[mask]
# P_QP = np.array(list([np.concatenate(QP_homo_P[str(cc_i)],axis=0) for cc_i in range(stop+1)]))[mask]


# In gerneral♌♌use the following
P_CD = list([np.concatenate(CD_homo_P[cc_i], axis=0) for cc_i in list(CD_homo_P.keys())])
P_SVD = list([np.concatenate(SVD_homo_P[cc_i], axis=0) for cc_i in list(SVD_homo_P.keys())])
P_QP = list([np.concatenate(QP_homo_P[cc_i], axis=0) for cc_i in list(QP_homo_P.keys())])

cluster_proteins = list(cluster_protein_name[cc_i] for cc_i in list(CD_homo_P.keys()))
print('The total number of connected_components is ⭃ {} \n Calculated number is ⭆ {}\n Skipped number is ⥅ {}'.format(cc_i,C-1,cc_i-C+1))

# ## 4. Calculate the feature between the inferred variables
# Cosin_QP_CD = np.array([hf.cosin2vector(P_CD[i],P_QP[j]) for i,j in zip(range(stop+1),range(stop+1))])
# Cosin_QP_SVD = np.array([hf.cosin2vector(P_SVD[i],P_QP[j]) for i,j in zip(range(stop+1),range(stop+1))])

## 4. Calculate the feature between the inferred variables♌♌
# avaible_CC = [int(i) for i in list(CD_homo_P.keys())]
avail_nC = len(list(CD_homo_P.keys()))
Cosin_QP_CD = np.array([hf.cosin2vector(P_CD[i],P_QP[j]) for i,j in zip(range(avail_nC), range(avail_nC))])
Cosin_QP_SVD = np.array([hf.cosin2vector(P_SVD[i],P_QP[j]) for i,j in zip(range(avail_nC), range(avail_nC))])

# Constructing feature origin for QP solver.
feature_origin_QP = np.array((P_QP_rsq, Neg_fract, Xi, CV_mean, CV_max, CV_min, X_corr_mean, P_eigen_sp, Cosin_QP_CD, Cosin_QP_SVD)).T
# feature_origin_QP



# More succinct Libraries structures for 3 Solvers
CD = dict({'homo_P': CD_homo_P, 'prot_sub':CD_prot_sub, 'opt':CD_opt})
QP = dict({'homo_P': QP_homo_P, 'prot_sub':QP_prot_sub, 'opt':QP_opt})

# test and check sovler solution
# sum(P_CD > 0)
# sum(P_QP > 0)
#
# import pandas as pd
# import HIquant_functions as hf
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# Corr_CD_QP = [np.corrcoef(P_CD[i],P_QP[j])[1][0] for i, j in zip(range(avail_nC),range(avail_nC))]
# Corr_SVD_QP = [np.corrcoef(P_SVD[i],P_QP[j])[1][0] for i, j in zip(range(avail_nC),range(avail_nC))]
# Corr_SVD_CD = [np.corrcoef(P_SVD[i],P_CD[j])[1][0] for i, j in zip(range(avail_nC),range(avail_nC))]
# # plt.hist([Corr_CD_QP, Corr_SVD_QP], color=['r','b'], alpha=0.5)
# corr_matrix = pd.DataFrame({'Corr: CD-QP':Corr_CD_QP, 'Corr: SVD-CD':Corr_SVD_CD, 'Corr: SVD-QP': Corr_SVD_QP})
# corr_matrix.plot(kind='hist', alpha = 0.6, bins = 20)
# plt.show()




# -----------*************************************************------------------------
#                     -----------Simulation-----------𝕴 | QP Solver | 𝕴--------
# -----------*************************************************------------------------
noise_set = hf.noise_set_gen(C)
level_noise_Number = noise_set.size
solvers = ['CD', 'QP', 'SVD']


# initialize a nest dictionary
P_inferred_expand_QP =collections.defaultdict(dict)
feature_expand_QP = collections.defaultdict(dict)
corr_QP = collections.defaultdict(dict)
overall_error_expand_QP = collections.defaultdict(dict)
ratio_error_expand_QP = collections.defaultdict(dict)
Cosinqp_CQ = collections.defaultdict(dict)
Cosinqp_SQ = collections.defaultdict(dict)

feature_expand_final_QP = dict()
# Loop over all connected_components for QP solver
for item in list(QP['homo_P'].keys()):
    # print(item)
    # print(' the connected_components {}'.format(item))
    prot_sub_noise_set = hf.protein_noise_set_gen(item, QP, noise_set)
    # simulation for each solver for each connected_components
    for each in solvers:
        # print('The solver {}'.format(each))
        noise_set_simulation = hf.P_inferred_expand_gen(prot_sub_noise_set, level_noise_Number, each)
        P_all, feature_expand, overall_error_expand, Median_ratio_expand, Corr_expand = noise_set_simulation
        # P_all dictionary to list of array
        P_all = list(P_all.values())
        P_inferred_expand_QP[str(each)][str(item)] = P_all
        feature_expand_QP[str(each)][str(item)] = feature_expand
        corr_QP[str(each)][str(item)] = Corr_expand
        overall_error_expand_QP[str(each)][str(item)] = overall_error_expand
        ratio_error_expand_QP[str(each)][str(item)] = Median_ratio_expand

    # Constructing the cosin_angle between two solver from above inference for all simulated noise levels
    for noise_level in range(level_noise_Number):
        CD_infer = np.concatenate(P_inferred_expand_QP['CD'][str(item)][noise_level])
        QP_infer = np.concatenate(P_inferred_expand_QP['QP'][str(item)][noise_level])
        SVD_infer = np.concatenate(P_inferred_expand_QP['SVD'][str(item)][noise_level])

        Cosinqp_CQ[str(item)][str(noise_level)] = hf.cosin2vector(CD_infer, QP_infer)
        Cosinqp_SQ[str(item)][str(noise_level)] = hf.cosin2vector(SVD_infer, QP_infer)

    # Constructing the additional expanded featues.
    f1 = np.array(list(Cosinqp_CQ[str(item)].values()))
    f2 = np.array(list(Cosinqp_SQ[str(item)].values()))
    f_additional = np.array((f1,f2)).T
    f_final = np.concatenate((feature_expand_QP['QP'][str(item)], f_additional),axis=1)
    feature_expand_final_QP[str(item)] = f_final
    #

# Return simulation feature and metric
simulation_features_QP = np.concatenate(list(feature_expand_final_QP.values()))
P_overall_QP = np.concatenate(list(overall_error_expand_QP['QP'].values()))
P_ratio_QP = np.concatenate(list(ratio_error_expand_QP['QP'].values()))
P_corr_QP = np.concatenate(list(corr_QP['QP'].values()))

# -----------*************************************************------------------------
#                     -----------Inference-----------𝕴 | QP Solver | 𝕴--------
# -----------*************************************************------------------------
# We will use random forest classifier

# traning - predicting - hist plotting
# overall_error
forest = RandomForestRegressor(300)
Model_oe = forest.fit(simulation_features_QP, P_overall_QP)
P_overall_QP_predict = Model_oe.predict(feature_origin_QP)
#pd.DataFrame(P_overall_QP_predict).plot(kind='hist', bins = 100)
#plt.xlim([0,1])  # Ed removed
#plt.show()	  # Ed removed

# Ratio_error
forest = RandomForestRegressor(300)
Model_re = forest.fit(simulation_features_QP, P_ratio_QP)
P_ratio_QP_predict = Model_re.predict(feature_origin_QP)
#pd.DataFrame(P_ratio_QP_predict).plot(kind='hist', bins = 100)
#plt.xlim([0,1])   # Ed removed
#plt.show()	   # Ed removed

# Correlation
forest = RandomForestRegressor(300)
Model_corr = forest.fit(simulation_features_QP, P_corr_QP)
P_corr_QP_predict = Model_corr.predict(feature_origin_QP)
#pd.DataFrame(P_corr_QP_predict).plot(kind='hist', bins = 100)
#plt.xlim([0,1]) # Ed removed
#plt.show() 	 # Ed removed


# save prediction.csv to desktop
fi = [dict(zip(cluster_protein_name[cc_i], QP_homo_P[cc_i])) for cc_i in  list(QP_homo_P.keys())]
from collections import ChainMap
#pd.DataFrame(dict(ChainMap(*fi))).T.to_csv('/Users/ed/Documents/GitHub/HIquant/Results/HIquant_predict_only_unmodi.csv')
pd.DataFrame(dict(ChainMap(*fi))).T.to_csv('/Users/ed/Documents/GitHub/HIquant/Results/Ed_PAICS.csv')
pd.DataFrame(dict(ChainMap(*fi))).T.to_csv(sys.argv[2])


# import pandas as pd
# pd.DataFrame(simulation_features_QP) #♌ last eigen spacing scale is different
# pd.DataFrame(simulation_features_QP).plot(kind='hist')
# pd.DataFrame({'Correlation': P_corr_QP, 'overall_error': P_overall_QP, 'Ratio_error': P_ratio_QP}).plot(kind='hist', bins = 20)

#pd.DataFrame({'Correlation': P_corr_QP_predict, 'overall_error': P_overall_QP_predict, 'Ratio_error': P_ratio_QP_predict}).T.to_csv('/Users/ed/Documents/GitHub/HIquant/Results/Ed_PAICSError2.csv')

pd.DataFrame({'Correlation': P_corr_QP_predict, 'overall_error': P_overall_QP_predict, 'Ratio_error': P_ratio_QP_predict}).T.to_csv(sys.argv[3])


print( 'HIquant inference complete')

#  Making plots
#sns.set(color_codes=True)
#sns.set_style("white")
#prediction = sns.distplot(P_overall_QP_predict, bins = 50, color="blue", label="Overall Error")
#prediction = sns.distplot(P_ratio_QP_predict, bins = 50, color="orange", label="Ratio Error")
#prediction = sns.distplot(P_corr_QP_predict, bins = 50, color="green", label="Correlation")
#plt.legend()
#plt.box(on=None)
#plt.title("HIquant Prediction_IL4(nf)")
#plt.show()
#figure = prediction.get_figure()
#figure.savefig('/Users/ed/Documents/GitHub/HIquant/Results/MCPtest.png', dpi =400)
