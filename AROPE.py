import numpy as np
import scipy.sparse.linalg as slin


def eigen_reweighting(X, order, coef):
#X: original eigenvalues
# order: order, -1 stands for infinity
# coef: weights, decaying constant if order = -1
    if order==-1:
        if len(coef)==1:
            if np.max(np.abs(X))*coef[0]<1:
                X_h = X/(1-coef[0]*X)
            else:
                print('Decaying constant is too large')
        else:
            print('Eigen reweighting wrong')
    elif len(coef)==order:
        X_h = coef[0]*X
        X_temp = X
        for i in range(1, order):
            #print('!')
            X_temp = X_temp*X
            X_h = X_h +coef[i]*X_temp
    else:
        print('Eigen reweighting wrong')
    return X_h    


def eigen_topL(A, d):
# A: N x N symmetric sparse adjacency matrix
# d: present dimension
# return top-L eigen-decomposition of A containing at least d positive eigenvalues
    if not np.allclose(A, np.transpose(A), atol=1e-8):
        print('The matrix is not symmetric!')
    L = d+10
    while 1:
        L+=d
        l, x = slin.eigs(A, k = L)
        if np.sum(l>0)>=d:
            break
    #select only top k
    inds = np.argsort(-np.absolute(l))
    max_ind = np.where(np.cumsum(l>0)>=d)
    l = l[:max_ind[0][0]]
    inds = inds[:max_ind[0][0]]
    x = x[:, inds]
    return l, x


def shift_embedding(lmbd, X, order, coef, d):
# lambda,X: top-L eigen-decomposition 
# order: a number indicating the order
# coef: a vector of length order, indicating the weights for each order
# d: preset embedding dimension
# return: content/context embedding vectors 
    lambda_h = eigen_reweighting(lmbd, order, coef)
    temp_index = np.argsort(-np.absolute(lambda_h))
    temp_index = temp_index[:d]
    lambda_h = lambda_h[temp_index]
    U = np.dot(X[:, temp_index], np.diag(np.sqrt(np.absolute(lambda_h))))
    V = np.dot(X[:, temp_index],  np.diag(np.sqrt(np.absolute(lambda_h))))*np.sign(lambda_h)

    return U, V


def AROPE(A, d, order, weights):
# AROPE Algortihm
# Inputs: 
# A: adjacency matrix A or its variations
# d: dimensionality 
# r different high-order proximity:
    # order: 1 x r vector, order of the proximity
    # weights: dictionary of r elements, each containing the weights for one high-order proximity
# Outputs: List of r elements, each containing the embedding Matrices     
    lambd, X = eigen_topL(A, d)
    r = len(order)
    U = []
    V = []
    for i in range(r):
        Ui, Vi = shift_embedding(lambd, X, order[i], weights[i], d)
        U.append(Ui)
        V.append(Vi)
    return U, V
