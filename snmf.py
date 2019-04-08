"""
NMF algorithms based on heuristic Multiplicative Update Rules.
- Beta-divergence 
- with/without sparseness constraints on the activation matrix. Several variants of constraints
- with/without dictionary normalization term in the Multiplicative Updates of W
- some chosen components of the dictionary are updated (unsupervised/semi-supervised/supervised NMF)

    Created on 04/2018
    Update on 01/2019

    @author: E.L. Benaroya - laurent.benaroya@gmail.com

    Copyright 2018 E.L. Benaroya
    This software is distributed under the terms of the GNU Public License
    version 3 (http://www.gnu.org/licenses/gpl.txt)
    
    TODO citer Le Roux!!!
    TODO citer Yosra Bekhti
    TODO modifier la fonction de cout (probleme avec LeRoux)

"""
import numpy as np
import itertools


def BetanmfSparse(X, W=1, indW=None, H=None, Beta=0., nbIter=100, noiseFloor=0.,
                  minVal=1e-16, Wupdate=True, Hupdate=True,
                  sparseType='None', Lambda=0., Lambda2=0., LRupdate=False):
    """
    NMf with Beta-divergence and some sparsity constraint on the activations
    either None, sum(l1/l2)^2 (horizontally or vertically computed norms), simple l1
    or l2,1 norm plus l1 norm.
    Uses classical heuristic Multiplicative Update Rules.
    An additive term to the MUR can be added to take into account the normalization of
    the dictionaries.

    Parameters
    ----------
    X : array, shape (F, N) - F = frequency bins, N = number of frames
        Matrix to factorize

    W : int or array, shape (F, K) - if W is an integer, then W is randomly drawn with K = W components
        template matrix or number of templates

    indW : vector (K1,) - indices of the dictionary components that will be updated

    H : array, shape(K,N) - activation matrix (optional)

    Beta : float, Beta-divergence parameter (default = 0., Itakura-Saito divergence)

    nbIter : int, number of iterations (default = 100)

    noiseFloor : float, added value to the data X and its approximation V to avoid numerical problems, especially with
    Itakura-Saito divergence

    minVal : float, minimum value - sort of equivalent to 'eps' in matlab :(

    Wupdate : bool, update the template matrix W if True (default = True)

    Hupdate : bool, update the activation matrix H if True (default = True)

    sparseType : string, type of sparsity constraint
        - 'none' no constraint (default)
        - 'l1' : classical l1 sparsity = Lambda * np.sum(H)
        - 'vsparse' : vertical (on columns) sparsity constraint (Normalized Norm L1)
        - 'hsparse' : horizontal (on lines) sparsity constraint (Normalized Norm L1)
        - 'l2hl1vsparse' : vertical sparsity with l1 norm (vertical) and l2 norm (horizontal)
            TODO : citer Yosra Bekhti

    Lambda : float, strength of the sparsity constraint (default = 0.)

    Lambda2 : in the l2,1 norm constraint ('l2hl1vsparse'), an additional l1 norm
    sparness constraint is be added, 'Lambda2' being the strength of this sparsness constraint.
    float (default = 0.)

    LRupdate (default False) : if True, use Le Roux et al. CITER !!!!! modified update rule
    on W before normalization. In this case, H is not re-scaled.

    Returns
    -------
    W : array, shape(F,K) - template matrix

    H : array, shape(K,N) - activation matrix

    V : array, shape(F,N) - approximation of X, V = np.dot(W,H)+noiseFloor

    Cost : vector, shape(nbIter,) - total cost at each iteration = Beta divergence D(X,V) + constraint cost C(H)
    """
    F, N = X.shape
    if isinstance(W, int):
        K = W
        W = np.random.rand(F, K)+minVal
        indW = range(K)
    else:
        K = W.shape[1]

    if indW is None:
        indW = range(K)

    if H is None:
        H = np.random.rand(K, N)+minVal
    indW
    X = X+noiseFloor
    Cost = np.zeros((nbIter,))

    # compute spectrogram approximation
    V = W.dot(H) + noiseFloor  # F x N

    spinner = itertools.cycle(['\\', '|', '/', '-'])

    for it in range(nbIter):
        it
        if Wupdate:
            numW = np.dot(X*np.power(V, Beta - 2), H[indW, :].T)  # F x K
            denW = np.dot(np.power(V, Beta - 1), H[indW, :].T)
            if LRupdate:
                numwlr = W[:, indW].dot(W[:, indW].T.dot(denW))
                denwlr = W[:, indW].dot(W[:, indW].T.dot(numW))
                W[:, indW] = W[:, indW]*(numW + numwlr + minVal)/(denW + denwlr + minVal)  # F x K
            else:
                W[:, indW] = W[:, indW]*(numW + minVal)/(denW + minVal)  # F x K
            W.shape
            # Dictionary normalization 
            sumW = np.sum(W[:, indW], axis=0).reshape((1, -1)) + minVal  # 1 x K
            W[:, indW] = W[:, indW] / sumW
            if not LRupdate:  # don't rescale H with LR update rule
                H[indW, :] = H[indW, :] * sumW.T

            # compute spectrogram approximation
            V = W.dot(H) + noiseFloor  # F x N
            V.shape
            H.shape
        if Hupdate:
            numH = np.dot(W.T, X*np.power(V, Beta - 2))  # K x N
            denH = np.dot(W.T, np.power(V, Beta - 1))

            constraintsCst = Lambda
            # if sparseType.lower() == 'none':
            numConstraint = 0.
            denConstraint = 0.
            distValue = 0.
            if sparseType.lower() == 'vsparse':
                currentMtx = H

                normL1_v = np.sum(np.abs(currentMtx), axis=0)
                normL2_v = np.sqrt(np.sum(np.abs(currentMtx)**2, axis=0)) + minVal
                n12 = normL1_v/(normL2_v**2)

                numConstraint = constraintsCst * 2.*currentMtx*(n12[np.newaxis, :]**2)
                denConstraint = constraintsCst * 2 * n12[np.newaxis, :]

                distValue = constraintsCst * np.sum((normL1_v/normL2_v)**2)
            elif sparseType.lower() == 'hsparse':
                currentMtx = H

                normL1_v = np.sum(np.abs(currentMtx), axis=1)
                normL2_v = np.sqrt(np.sum(np.abs(currentMtx) ** 2, axis=1)) + minVal
                n12 = normL1_v / (normL2_v ** 2)
                # n12 = n12[:, np.newaxis]
                numConstraint = constraintsCst * 2. * currentMtx * (n12[:, np.newaxis] ** 2)
                denConstraint = constraintsCst * 2 * n12[:, np.newaxis]

                distValue = constraintsCst * np.sum((normL1_v / normL2_v) ** 2)
            elif sparseType.lower() == 'l1':
                numConstraint = 0.
                denConstraint = constraintsCst
                distValue = constraintsCst * np.sum(H)
            elif sparseType.lower() == 'l2h1vsparse':
                # Lambda2 is the l1 norm constraint strength
                l2h = np.sqrt(np.sum(H**2, axis=1))
                numConstraint = 0.
                denConstraint = constraintsCst * H / l2h[:, np.newaxis] + Lambda2
                distValue = constraintsCst * np.sum(l2h) + Lambda2 * np.sum(H)
            H = H * (numH + numConstraint + minVal) / (denH + denConstraint + minVal)  # F x N
            # H[H<minVal] = minVal
            # compute spectrogram approximation
            V = W.dot(H) + noiseFloor  # F x N

        # compute cost fct

        if Beta == 0:  # Itakura-Saito
            C = X/V - np.log(X/V) - 1
        elif Beta == 1:  # KL div.
            C = X*np.log(X/V+minVal) + V - X
        elif Beta == 2:  # euclidean
            C = 0.5 * np.power(X-V, 2)
        else:  # general case
            C = 1./(Beta*(Beta-1))*(np.power(X, Beta) + (Beta-1)*np.power(V, Beta)
                                    - Beta*X*np.power(V, Beta-1))

        Cost[it] = np.sum(C)+distValue
        if it % 5 == 0:
            print('%s It %d, cost : %.2f' % (next(spinner), it, Cost[it]), end="\r")
        
    print("")
    return W, H, V, Cost
