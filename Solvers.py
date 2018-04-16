import numpy as np
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False
from scipy.linalg import block_diag, svd
import sys
import HIquant_functions as hf


def SVD(prot_sub):
    """
    SVD sovler:
    Return the inferred protein level, Z, r^2
    """
    S = prot_sub['S']
    X = prot_sub['X']
    N = X.shape[1]
    M = S.shape[0]
    K = S.shape[1]
    A1 = np.kron(S, np.eye(N))
    A2 = block_diag(*X).T
    A = np.concatenate((A1, -A2), axis=1)
    U, s, V = np.linalg.svd(A, full_matrices=True)
    # U, s, V = svd(A, full_matrices=True, lapack_driver = 'gesvd')
    w = V[-1,:]#♌
    eigen_spacing = (s[-2]-s[-1])/(s[-2]+s[-1])
    if np.sum(w < 0) > np.sum(w > 0):
        w = -1 * w
    p_vect = w[0: K * N]
    U_hat = p_vect.reshape(K, N)
    L_hat = w[K * N::]
    Z_hat = 1/L_hat
    protein = U_hat

    X_rec = np.diag(Z_hat).dot(S).dot(U_hat)
    Model = X_rec * np.median(X_rec[:]/X[:])
    X_norm = hf.norm_mean(X,0)
    Rsq = 1- (np.sum(X[:] - Model[:])**2)/np.sum(X_norm**2)

    opt = dict()
    opt['model'] = Model
    opt['rsq'] = Rsq
    opt['Z'] = np.diag(Z_hat)
    opt['V'] = w # last eigen vector
    opt['eigen_spacing'] = eigen_spacing
    return protein, opt



def QP(prot_sub):
    """
    Quadratic Programming Solver:
    Return the inferred protein level, Z, r^2
    """
    S = prot_sub['S']
    X = prot_sub['X']
    N = X.shape[1]
    A1 = np.kron(S, np.eye(N))
    A2 = -block_diag(*X).T
    A1 = matrix(A1)
    A2 = matrix(A2)
    AA = matrix([[A1], [A2]])
    P = matrix(np.dot(AA.T, AA))
    dim = P.size[0]
    q = matrix(np.zeros(dim))
    G = matrix(-np.eye(dim))
    h = matrix(np.zeros(dim), (dim, 1))
    A = matrix(np.ones(dim), (1, dim))
    b = matrix(1.0)
    sol = solvers.qp(P, q, G, h, A, b)
    w = np.array(sol['x'])

    if np.sum(w < 0) > np.sum(w > 0):
        w = -1 * w
    p_vect = w[0:S.shape[1]*X.shape[1]]
    U_hat = np.array(p_vect).reshape(S.shape[1],X.shape[1]) #♌ in test case reshape(2,6)
    L_hat = np.concatenate(w[S.shape[1]*X.shape[1]::])
    Z_hat = 1/L_hat
    protein = U_hat

    X_rec = np.diag(Z_hat).dot(S).dot(U_hat)
    Model = X_rec * np.median(X_rec[:]/X[:])
    X_norm = hf.norm_mean(X,0)
    Rsq = 1- (np.sum(X[:] - Model[:])**2)/np.sum(X_norm**2)

    opt = dict()
    opt['model'] = Model
    opt['rsq'] = Rsq
    opt['Z'] = np.diag(Z_hat)
    return protein, opt



def CD(prot_sub, iter_num):
    """
    Coordinate Descent Solver:
    retrun inferred protein, Z, r^2, X, error
    """
    S = prot_sub['S']
    X = prot_sub['X']

    if len(sys.argv) < 3:
        iter_num = 1000

    Num_pep, Num_prot = S.shape
    Num_condition = X.shape[1]

    # set IC
    protein_ic = np.random.randn(Num_prot,Num_condition)
    protein = protein_ic

    # for loop iterations
    for each in range(iter_num):
        Prot = protein
        dat = S.dot(protein).T
        zeta = np.sum(dat * X.T,axis=0)/np.sum(dat*dat,axis=0)
        if np.sum(zeta[:]<0):
            zeta[zeta<0] = 1e-4
        Zeta = np.diag(zeta)

        Proteins = np.linalg.lstsq(Zeta.dot(S), X)[0]

        if np.sum(Proteins[:]<0):
            Proteins[Proteins<0] = 1e-4
        protein = Proteins

        if np.linalg.norm(Prot[:] - protein[:]) < 1e-8:
            break
    Model = Zeta.dot(S).dot(protein)
    X_norm = hf.norm_mean(X,0)
    Rsq = 1- (np.sum(X[:] - Model[:])**2)/np.sum(X_norm**2)

    opt = dict()
    opt['model'] = Model
    opt['rsq'] = Rsq
    opt['Z'] = Zeta
    opt['error'] = np.linalg.norm(Prot[:] - protein[:])
    return protein, opt
