import numpy as np
import numpy.matlib as mtl
from cvxopt import solvers, matrix
from itertools import combinations
from scipy.linalg import block_diag
import sys
import random
import pandas as pd
import scipy.sparse as sps
from scipy.sparse.csgraph import connected_components
import Solvers
import numba
@numba.jit

def norm_mean(matrix, dim):
    """
    normalized each row by subtracting the mean of the row
    if dim = 0, matrix is normalized along row dimension which subtracting each row by its row mean;
    if dim = 1, matrix is normalized along column dimension which subtracting each column by its column mean
    """
    if dim == 0:
        matrix = matrix - np.mean(matrix,axis=1,keepdims=True)
    elif dim == 1:
        matrix = matrix - np.mean(matrix,axis=0,keepdims=True)
    return matrix


def norm_div(matrix, dim):
    """
    normalized each row by dividing the mean of the row
    if dim = 0, matrix is normalized along row dimension which dividing each row by its row mean;
    if dim = 1, matrix is normalized along column dimension which dividing each column by its column mean
    """
    if dim == 0:
        matrix = matrix /np.mean(matrix,axis=1,keepdims=True)
    elif dim == 1:
        matrix = matrix /np.mean(matrix,axis=0,keepdims=True)
    return matrix




def cosin2vector(a, b):
    """
    return the coisin angle between vactor a & vector b
    """
    Dot_ab = np.dot(a,b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosin_angle = Dot_ab/(norm_a*norm_b)
    return cosin_angle




def cv_score(matrix, case):
    """
    The Coeficient of varitaion calcultes the σ/μ in a vectorized way along row dimension
    input case options: 1) rows cv mean 2) rows cv max 3) rows cv min
    """
    cv = np.std(matrix,axis=1,ddof=1)/np.mean(matrix,axis=1) # the std in matlab is not exact = np.std
    if case == 'mean':
        cv = np.mean(cv)
    elif case == 'max':
        cv = np.max(cv)
    elif case == 'min':
        cv = np.min(cv)
    return cv



def null_sp_dim(prot_sub):
    """
    The null_sp_dim function return the null space dimension of given S matrx.
    ℜ: default number of condition is 6, if different, please specific to *args
    """
    S = prot_sub['S']
    N = prot_sub['X'].shape[1]
    M,K = S.shape

    Z_true = 1 + np.array([random.expovariate(1) for rand in range(M)])
    U_true = np.random.rand(K, N)
    X = np.diag(Z_true).dot(S).dot(U_true)

    A1 = np.kron(S, np.eye(N))
    A2 = block_diag(*X).T
    A = np.concatenate((A1, -A2), axis=1)
    U, s, V = np.linalg.svd(A, full_matrices=True)
    col_dim = sum( s >1e-10 )
    num_dim = A.shape[1]-col_dim
    return num_dim



def mean_squareform(corr_m):
    """
    Take the upper triangle non-diagnal elements into a vector
    """
    dim = corr_m.shape[0]
    accumulate_corr = np.array([])
    for idx in range(len(corr_m)):
        row_values = corr_m[idx, idx+1::]
        accumulate_corr = np.append(accumulate_corr, row_values)
    return np.mean(accumulate_corr)




def mSX2_notright(S,X):
    """
    Function that colapses the data for peptides mapping to the same set of
    proteins to a single (median) estimate
    """
#     # find the peptide indices belong to the same protein.
#     prot_multipep_id = np.where(np.nansum(prot_S,0) > 1)[0]
    a, idd = np.unique(S,axis=0,return_index=True)
#     print('unique rows: \n',a)
#     print('unique row index: ',idd)

    mask = ~np.in1d(np.arange(len(S)),idd)
#     print('The non-unique row index:',np.where(mask==True)[0])
#     print('The duplicated peptide rows:\n ',S[mask])


    X_collapse = np.zeros((len(a),X.shape[1]))
    uniq_row_set = np.array([],dtype=int)
    # loop over all unique rows
    for i in range(len(S[mask])):
#         print('The duplicated S row: ',S[mask][i])
        duplicate_row_index = np.array(np.where(mask==True)[0][i])
#         print('Duplicated row index:', duplicate_row_index)
        a, idd = np.unique(S,axis=0,return_index=True)
        match = [np.array_equal(S[mask][i], each_uniq) for each_uniq in np.unique(S,axis=0)]
    #     print('matched row in the unique index set is ', match)
        uniq_row_id = idd[match]
#         print('The matched unique set row index is ', uniq_row_id)
        uniq_row_set = np.append(uniq_row_set,uniq_row_id)

        # collapse the X data matrix:
        # for each duplicated rows:
        select_id = np.append(uniq_row_id,duplicate_row_index)
        X_collapse_i = X[select_id]
        row_norm = X_collapse_i/np.mean(X_collapse_i,axis=1,keepdims=True)
        rows_collapse = np.median(row_norm,axis=0)
#         print('The collapsed normalized row:',rows_collapse)
        # assign row collapse values to the unique index row in X
        X_collapse[uniq_row_id,:] = rows_collapse


    uniq_left_row = np.setdiff1d(idd,uniq_row_set)
#     print('The unique row index that does not have duplicates is:',uniq_left_row)
    X_collapse[uniq_left_row,:] = X[uniq_left_row,:]

    X_collapse =X_collapse[idd]
    S_collapse = S[idd]
    return S_collapse, X_collapse



def mSX2(S,X):
    """
    Function that colapses the data for peptides mapping to the same set of
    proteins to a single (median) estimate
    """
    uniq_r = np.unique(S, axis=0)
    X_collapse = dict()

    for each in uniq_r:
        # print(each)
        # return mask for each uniq_r in S
        mask = [np.array_equal(each,S[i,:]) for i in range(S.shape[0])]
        # print(mask)
        # return the corresponding rows in X and collapse them
        # print('original X',X)
        # print('Sub_selected X',X[mask])
        X_select_each = X[mask]
        row_norm = X_select_each/np.mean(X_select_each,axis=1,keepdims=True)
        X_collapse_i = np.median(row_norm,axis=0)
        # X_collapse = np.append(X_collapse, X_collapse_i)
        X_collapse[str(each)] = X_collapse_i
        # print('collapse X \n',X_collapse)

    X_collapse = np.array(list(X_collapse.values()))
    S_collapse = uniq_r

    return S_collapse, X_collapse




def GetData (path):
    """
    Parsing the peptide level data:
    return the following variables
    ▶: Mat_A , Mat_S, Mat_CC , Mat_nC, Dat_data, Dat_textdata, Dat_proteins, Data_isunique, Num_peptides, Num_proteins
    """
    if len(sys.argv) < 1:
        print (' You have to input the absolute Path of your file! ')
    else:
        Path=path
        print( 'Importing data from the directory below ...')
        print ('The file path you inputed is : \n', Path)
        dat = pd.read_csv(Path,sep='\t',header=None)
        dat = dat.dropna()
        Textdata = dat.iloc[:,[0,1]]
        Data = dat.iloc[:,2::]
        #Data = Data/np.mean(Data, axis=0)
        Data_cn = Data.apply(lambda x : x/x.mean(), axis = 0)# normalize column by its mean
        Data_rn = Data_cn.apply(lambda x : x/x.mean(), axis = 1)# normalize row by its mean
        Data_norm = Data_rn
        # Proteins = dat.iloc[:,0].str.split(';', expand=True).iloc[:, 0].unique()
        all_protein_str = Textdata.iloc[:,0].str.split(';',expand = True).values
        Proteins = np.unique(all_protein_str[all_protein_str != None])
        # Check the number of Columns, HIquant need at least 2 columns
        if len(Data.columns) < 2:
            print('HIquant needs at least 2 conditions')
            return
        else:
            print('Number of conditions checked !')

        # find the unique number of proteins
        Num_proteins = len(np.unique(Proteins))
        # find the number of peptides
        Num_peptides = len(dat.index)

        # Making Stochiometry Matrix （wrong）
        print('Constructing Stochiometry Matrix')
        A = sps.lil_matrix((Num_proteins,Num_proteins))
        S = np.zeros((Num_peptides, Num_proteins))
        Data_isunique = np.zeros((Num_peptides,1))
        for pep_i in range(Num_peptides):
            proteins = Textdata.iloc[pep_i,0].split(';')
            num_prots = len(Textdata.iloc[pep_i,0].split(';'))
            idxlist = []
#             print('The peptide {} associates {} proteins : {}'.format(pep_i,num_prots,proteins))
            for prot_i in range(num_prots):
                p = proteins[prot_i]
                indices, =np.where(np.in1d(Proteins,p))
                S[pep_i,indices] = S[pep_i,indices] + 1
#                 print('idxlist is :', idxlist)
#                 print('indices is :', indices)
                idxlist.append(indices)#[prot_i]=indices
#                 print('The protein {} has associated index {}'.format(p,indices))
#                 print('The indices {} in the orignal Proteins list is protein:{}'.format(indices,Proteins[indices]))
#                 print('The S matrix row index:{} column index:{}'.format(pep_i, indices))
            idxlist = np.unique (idxlist)
#             print('The peptide {} final protein column index list:{}\n'.format(pep_i, idxlist))
            Data_isunique[pep_i] = len(idxlist)
            newid = []
            for i in idxlist:
                for j in idxlist:
                    newid.append([i,j])
            newid = np.array(newid)
            A[newid[:, 0], newid[:, 1]] = 1


        print ('Stochi matrix finished!')
        n_components, component_list = connected_components(A,directed=True)

        # Returning the following values
        Mat_A = A
        Mat_S = S
        Mat_CC = component_list
        Mat_nC = n_components
        Dat_data = Data_norm
        Dat_textdata = Textdata
        Dat_proteins = Proteins
        return Mat_A , Mat_S, Mat_CC , Mat_nC, Dat_data, Dat_textdata, Dat_proteins, Data_isunique, Num_peptides, Num_proteins





def noise_set_gen(nC):
    """
    In order to have sufficient number of data points for training the model,
    we need to have a certain nosie_set that will give a a final > 1000 dat points
    given the number of cluster (nC).
    """
    # 1). Each input need at least 1000 simulated data points.
    cluster_number = nC
    potential_noise = 1000/cluster_number
    # 2. Comparing potential noise levels with 6 base levels, and take the max number to further calculated how many duplicates need to be added to each noise levels.
    select = np.array([6, potential_noise]).max()
    select
    # 3. Determin how many duplicated for each noise level
    sigma_n = 0.3
    if select > 6:
        ns = np.ceil(select/6)
        noise_set = np.linspace(0.05, sigma_n, 6)
        noise_set = mtl.repmat(noise_set, 1, int(ns))[0]
    else:
        noise_set = np.linspace(0.05, sigma_n, 6)
    return noise_set




def protein_noise_set_gen(each_cluster, solver, noise_set):
    """
    For give solver and the noise_set,
    it will generate a simulated dat dictionary for each noise level
    """
    prot_sub_noise_set = np.array([])
    for r in range(noise_set.size):
        P = solver['homo_P'][str(each_cluster)]
        S = solver['prot_sub'][str(each_cluster)]['S']
        Z_true = solver['opt'][str(each_cluster)]['Z']
        X = Z_true.dot(S).dot(P)
        X = X + noise_set[r]*X*np.array(np.matlib.randn(X.shape))
        prot_sub = dict({'X': X, 'S': S, 'Z': Z_true, 'P': P})
        prot_sub_noise_set = np.append(prot_sub_noise_set,prot_sub)
    return prot_sub_noise_set




def P_inferred_expand_gen(prot_sub_noise_set, level_noise_Number, solver):
    # global each
    # each of following is list of array for each CC
    P = [prot_sub_noise_set[i]['P'] for i in range(level_noise_Number)] # same for all noise
    Z_true = [prot_sub_noise_set[i]['Z'] for i in range(level_noise_Number)] # same for all noise
    S = [prot_sub_noise_set[i]['S'] for i in range(level_noise_Number)] # same for all noise
    X = [prot_sub_noise_set[i]['X'] for i in range(level_noise_Number)] # different for each noise

    # U_hat = np.zeros(P[0].shape)
    P_all = dict()
    # R2_expand = dict()
    # P_all = np.array([])
    R2_expand = np.array([])
    overall_error_expand = np.array([])
    Median_ratio_expand = np.array([])
    Corr_expand = np.array([])
    Xi = np.array([])
    CV_mean = np.array([])
    CV_min = np.array([])
    CV_max = np.array([])
    X_corr_mean = np.array([])
    lst_sp_expand = np.array([])
    Neg_fract_expand = np.array([])

    for r in range(level_noise_Number):
        # print(' the {} noise level'.format(r))
        prot_sub = prot_sub_noise_set[r]
        if solver == 'QP':
            QP_solution = Solvers.QP(prot_sub)
            protein_inferred, opt = QP_solution

        elif solver == 'CD':
            CD_solution = Solvers.CD(prot_sub, 1000)
            protein_inferred, opt = CD_solution

        elif solver == 'SVD':
            SVD_solution = Solvers.SVD(prot_sub)
            protein_inferred, opt = SVD_solution

        alpha_SC = np.median(np.diag(Z_true[0])/np.diag(opt['Z']))# rescale to match it closely (unidentifiable up-to-const)
        U_hat = protein_inferred
        U_hat = U_hat * (1/alpha_SC)

        # -----Calculate Feature associated with each U_hat-----
        # 1) R^2
        rsq = opt['rsq']
        R2_expand = np.append(R2_expand, rsq)
        # R2_expand[str(r)] = rsq
        # R2_expand_t.append(rsq)

        # 2) Overall Error (♌ some of the element in U is zero which end up with divide by zeros element)
        rescale = np.median(np.concatenate(P[r])/np.concatenate(U_hat))
        P_scaled = U_hat * rescale
        overall_error = np.median(np.abs(P_scaled - P[r])/P[r])
        overall_error_expand = np.append(overall_error_expand, overall_error)
        # 3) Ratio Error
        #3.1 all possible combinations of rows of P
        combo = list(combinations(list(range(P[r].shape[0])),2))
        ratio_set = np.array([])
        for pair in combo:
            row1, row2 = pair
            R_old = P[r][row1,:]/ P[r][row2,:]
            R_new = U_hat[row1,:]/ U_hat[row2,:]
            R_i = (R_new - R_old)/ R_old
            ratio = np.median(np.abs(R_i))
            ratio_set = np.append(ratio_set, ratio)
        Median_ratio_i = np.median(ratio_set)
        Median_ratio_expand = np.append(Median_ratio_expand, Median_ratio_i)
        # print('The {} noise level has {} ratio set '.format(r, ratio_set))
        # 4) correlation (U_hat, P)
        Corr_P_P_true = np.corrcoef(np.concatenate(P[r]), np.concatenate(U_hat))[1][0]
        Corr_expand = np.append(Corr_expand, Corr_P_P_true)

        # 5) Xi
        P_relative_norm_temp = np.std(prot_sub['X'],axis=1) / np.mean(prot_sub['X'],axis=1)
        P_relative_norm = np.nanmean(P_relative_norm_temp)
        Xi = np.append(Xi, P_relative_norm)

        # 6) CV score for each simulated P
        cv_mean_i = cv_score(U_hat, 'mean')
        cv_max_i = cv_score(U_hat, 'max')
        cv_min_i = cv_score(U_hat, 'min')
        CV_mean = np.append(CV_mean, cv_mean_i)
        CV_max = np.append(CV_max, cv_max_i)
        CV_min = np.append(CV_min, cv_min_i)
        # 7) The mean correlation between the rows of X as a measure of linear independence between the the columns of X
        X_corr = np.corrcoef(prot_sub['X'].T)
        X_corr_modi = X_corr - np.eye(X_corr.shape[0]) # the diagnal of X_corr shows 1, but when subtract by np.eye, the result shows a close to zeros Number
        X_corr_mean_i = mean_squareform(X_corr_modi)
        X_corr_mean = np.append(X_corr_mean, X_corr_mean_i)

        # 8) last eigen_spacing
        _, opt_eigen = Solvers.SVD(prot_sub)
        lst_sp_expand = np.append(lst_sp_expand, opt_eigen['eigen_spacing'])

        # 9) calculate the fraction of negative elements in the smallest singular vector of A, when v1 = median(sign(v1))v1
        vi = opt_eigen['V']
        vi = np.median(np.sign(vi))*vi
        Neg_fract_i = np.sum(vi<0)/len(vi)
        Neg_fract_expand = np.append(Neg_fract_expand, Neg_fract_i)

        # make P_inferred dictionary for the noise set.
        P_all[str(r)] = U_hat
        # P_all = np.append(P_all, U_hat)

    # Constructing the expanded feature Matrix
    feature_expand = np.array((R2_expand, Neg_fract_expand, Xi, CV_mean, CV_max, CV_min, X_corr_mean, lst_sp_expand)).T
    return P_all, feature_expand, overall_error_expand, Median_ratio_expand, Corr_expand
