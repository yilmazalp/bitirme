# -*- coding: utf-8 -*-

import numpy as np 

def asRowMatrix (X):
    if len (X) == 0:
        return np. array ([])
    mat = np. empty ((0 , X [0]. size ), dtype =X [0]. dtype )
    
    for row in X:
        mat = np. vstack (( mat , np. asarray ( row ). reshape (1 , -1)))
    return mat


def asColumnMatrix (X):
    if len (X) == 0:
        return np. array ([])
        
    mat = np. empty ((X [0]. size , 0) , dtype =X [0]. dtype )
    
    for col in X:
        mat = np. hstack (( mat , np. asarray ( col ). reshape ( -1 ,1)))
    return mat
    

def pca (W, y, num_components = 0):
    [n,d] = W.shape
    
    if (num_components <= 0) or (num_components >n):
        num_components = n
        
    mu = W.mean( axis =0)
    W = W - mu
    
    if n>d:
        C = np.dot(W.T,W)
        [eigenvalues,eigenvectors] = np.linalg.eigh(C)
        
    else :
        C = np.dot(W,W.T)
        [eigenvalues , eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(W.T, eigenvectors)
        
    for i in range(n):
        eigenvectors[:,i] = eigenvectors[:,i]/ np.linalg.norm(eigenvectors[:,i])
        
    # or simply perform an economy size decomposition
    # özvektörler, özdeğerler, varyans = np. linalg . svd (X.T, full_matrices = False )
    # özvektörleri özdeğerlerin azalan sırasına göre sıralamak
        
    idx = np. argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    
    # sadece num_components değişkenlerini seçmek
    eigenvalues = eigenvalues[0: num_components].copy()
    eigenvectors = eigenvectors [: ,0: num_components ].copy()
    return [eigenvalues, eigenvectors, mu]
    
    
    
    
    
def project (W, X, mu= None):
    if mu is None:
        return np.dot(X,W)
    return np.dot(X - mu, W)
    
    
    
    
    
    
    
    
    
    
    
def reconstruct (W, Y, mu= None):
    if mu is None :
        return np.dot(Y, W.T)
    return np.dot(Y, W.T) + mu









    