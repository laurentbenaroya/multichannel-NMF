# -*- coding: utf-8 -*-
"""
Frobenius loss function
Use projected gradient with Wolfe line search
Optimisation class definition
"""

# Author : E.L. Benaroya - laurent.benaroya@gmail.com
# Date : 04/2019
# License : GNU GPL v3

import numpy as np
# import time


class FroMnmfOptim:

    def __init__(self, X, A, W, H, sigmab, num_comp=10, CovX=None, set_cov_real=True):
        """

        Parameters
        ----------
        X
        A
        W
        H
        sigmab
        num_comp
        CovX
        set_cov_real

        Attributes
        ----------

        """
        self._X = X
        self._A = A
        if A.ndim == 2:
            self._mixtype = 'inst'
        elif A.ndim == 3:
            self._mixtype = 'conv'
        else:
            raise Exception('The mixing matrix must have two or three dimensions')

        self._W = W
        self._H = H
        self._sigmab = sigmab

        # num_comp can be an int or a np array with the number of nmf cmp per source
        J = A.shape[1]
        if isinstance(num_comp, int):
            num_comp = num_comp*np.ones((J, ), dtype=np.int)

        # total number of nmf components
        n_comp_tot = np.sum(num_comp)

        # create indices dictionary
        num_comp = np.hstack((0, num_comp))
        num_comp = np.cumsum(num_comp)
        comp_dict = {j: np.arange(num_comp[j], num_comp[j+1], dtype=int) for j in range(J)}

        self._comp_dict = comp_dict
        self._n_comp_tot = n_comp_tot

        self._cost = np.inf
        self._grad_x = 0.
        self._grad_W = 0.
        self._grad_H = 0.
        self._grad_A = 0.
        self.cost = 0.

        F, N, I = X.shape
        J = A.shape[1]
        # compute empirical covariance matrix
        if CovX is not None:
            self._XX = CovX
        else:
            XXh = np.zeros((F, N, I, I), dtype=np.complex)
            for f in range(F):
                for n in range(N):
                    x = X[f, n, :]
                    XXh[f, n, :, :] = np.outer(x, x.conj())
            self._XX = XXh
        self._sources = 0.  # np.zeros((F, N, J), dtype=np.complex)

        if set_cov_real:
            self._XX = np.real(self._XX)

    def froCost(self, A=None, W=None, H=None, sigmab=None,
                grad=None, dosave=True, output=True, verbose=False):
        """

        Parameters
        ----------
        A
        W
        H
        sigmab
        grad
        dosave
        output
        verbose

        Returns
        -------

        """
        if A is None:
            A = self._A
        if W is None:
            W = self._W
        if H is None:
            H = self._H
        if sigmab is None:
            sigmab = self._sigmab

        n_comp_tot = self._n_comp_tot

        F, N, I = self._XX.shape[:3]  # (F x N x I x I)
        J = A.shape[1]  # A (I x J)

        sigmax = np.zeros((F, N, I, I), dtype=np.complex)

        V = np.zeros((F, N, J))
        # n_comp = self._n_comp
        for j in range(J):
            Wj = W[:, self._comp_dict[j]]
            Hj = H[self._comp_dict[j], :]
            Vj = Wj.dot(Hj)
            V[:, :, j] = Vj  # variance per source

        sigmax[:, :, 0, 0] = sigmab
        sigmax[:, :, 1, 1] = sigmab

        for j in range(J):
            sigmax[:, :, 0, 0] += (np.abs(A[0, j]) ** 2) * V[:, :, j]
            sigmax[:, :, 0, 1] += (A[0, j] * A[1, j].conj()) * V[:, :, j]
            sigmax[:, :, 1, 1] += (np.abs(A[1, j]) ** 2) * V[:, :, j]
        sigmax[:, :, 1, 0] = sigmax[:, :, 0, 1].conj()

        cost = 0.5*np.linalg.norm((sigmax-self._XX).flatten(), ord=None)**2
        if verbose:
            print('Cost: %.3f' % cost)

        if dosave:
            self._cost = cost

        if grad is not None:
            grad_x = sigmax-self._XX  # F x N x I x I

        if dosave:
            self._grad_x = grad_x

        if grad == 'W' or grad == 'H':

            grad_sigmass = np.zeros((F, N, J), dtype=np.complex)
            for j in range(J):
                grad_sigmass[:, :, j] = \
                    A[0, j] * grad_x[:, :, 0, 0] * A[0, j].conj() + \
                    A[0, j] * grad_x[:, :, 0, 1] * A[1, j].conj() + \
                    A[1, j] * grad_x[:, :, 1, 0] * A[0, j].conj() + \
                    A[1, j] * grad_x[:, :, 1, 1] * A[1, j].conj()
            n_comp_tot = self._n_comp_tot

        if grad == 'W':

            grad_W = np.zeros((F, n_comp_tot), dtype=np.float)
            for j in range(J):
                indjj = self._comp_dict[j]
                for jj in indjj:
                    grad_W[:, jj] = np.real(np.sum(grad_sigmass[:, :, j] *
                                                   self._H[jj, :].reshape((1, -1)),
                                                   axis=1))
            if dosave:
                self._grad_W = grad_W

        elif grad == 'H':

            grad_H = np.zeros((n_comp_tot, N), dtype=np.float)
            for j in range(J):
                indjj = self._comp_dict[j]
                for jj in indjj:
                    grad_H[jj, :] = np.real(np.sum(grad_sigmass[:, :, j] *
                                                   self._W[:, jj].reshape((-1, 1)),
                                                   axis=0))
            if dosave:
                self._grad_H = grad_H

        elif grad == 'A':

            grad_A = np.zeros_like(A)
            for j in range(J):
                grad_A[0, j] = np.sum((grad_x[:, :, 0, 0]*A[0, j]+grad_x[:, :, 0, 1]*A[1, j]) *
                                      V[:, :, j])
                grad_A[1, j] = np.sum((grad_x[:, :, 1, 0]*A[0, j]+grad_x[:, :, 1, 1]*A[1, j]) *
                                      V[:, :, j])
            grad_A *= 2
            if dosave:
                self._grad_A = grad_A

        if output:
            if grad == 'W':
                return cost, grad_W
            elif grad == 'H':
                return cost, grad_H
            elif grad == 'A':
                return cost, grad_A
            else:
                return cost

    def gradentdescent(self, gradtype, iter=1, dosave=False, verbose=False):
        """

        Parameters
        ----------
        gradtype
        iter
        dosave
        verbose

        Returns
        -------

        """
        # update W
        if gradtype == 'W':
            if verbose:
                print('Updating W')
            W = self._W
            fk, gradW = self.froCost(grad='W', dosave=False)
            m = -np.sum(gradW ** 2)
            a = 0.
            b = + np.inf
            t = .1

            epsilon1 = 0.0001
            epsilon2 = 0.9

            w_min = 1e-10

            # Wolfe
            for line_ls in range(100):
                Wk_ls = np.maximum(W - t * gradW, w_min)
                self._W = Wk_ls
                fk_ls = self.froCost(grad='None', dosave=False)

                if fk_ls > fk + t * epsilon1 * m:
                    b = t
                    t = 0.5 * (a + b)
                else:
                    fkk_ls, gradW_ls = self.froCost(grad='W', dosave=False)
                    if -np.sum(gradW * gradW_ls) < epsilon2 * m:
                        a = t
                        if b == +np.inf:
                            t = 2 * a
                        else:
                            t = 0.5 * (a + b)
                    else:
                        break
            if verbose:
                print("step size for W = %d", t)
            if not dosave:
                self._W = W
                return Wk_ls
            else:
                self._W = Wk_ls

        # update H
        if gradtype == 'H':
            if verbose:
                print('Updating H')
            H = self._H
            fk, gradH = self.froCost(grad='H', dosave=False)
            m = -np.sum(gradH ** 2)
            a = 0.
            b = 3.  # + np.inf
            t = .1

            epsilon1 = 0.0001
            epsilon2 = 0.9

            h_min = 1e-10

            # Wolfe
            for line_ls in range(100):
                Hk_ls = np.maximum(H - t * gradH, h_min)
                self._H = Hk_ls
                fk_ls = self.froCost(grad='None', dosave=False)

                if fk_ls > fk + t * epsilon1 * m:
                    b = t
                    t = 0.5 * (a + b)
                else:
                    # self._W = Hk_ls
                    fkk_ls, gradH_ls = self.froCost(grad='H', dosave=False)
                    if -np.sum(gradH * gradH_ls) < epsilon2 * m:
                        a = t
                        if b == +np.inf:
                            t = 2 * a
                        else:
                            t = 0.5 * (a + b)
                    else:
                        break
            if verbose:
                print("step size for H = %d", t)
            if not dosave:
                self._H = H
                return Hk_ls
            else:
                self._H = Hk_ls

        # update A
        if gradtype == 'A':
            if verbose:
                print('Updating A')
            A = self._A
            fk, gradA = self.froCost(grad='A', dosave=False)
            m = -np.sum(gradA ** 2)
            a = 0.
            b = 3.  # + np.inf
            t = .1

            epsilon1 = 0.0001
            epsilon2 = 0.9

            a_min = 1e-10

            # Wolfe
            for line_ls in range(100):
                Ak_ls = np.maximum(A - t * gradA, a_min)
                self._A = Ak_ls
                fk_ls = self.froCost(grad='None', dosave=False)

                if fk_ls > fk + t * epsilon1 * m:
                    b = t
                    t = 0.5 * (a + b)
                else:
                    # self._W = Hk_ls
                    fkk_ls, gradA_ls = self.froCost(grad='A', dosave=False)
                    if -np.sum(gradA * gradA_ls) < epsilon2 * m:
                        a = t
                        if b == +np.inf:
                            t = 2 * a
                        else:
                            t = 0.5 * (a + b)
                    else:
                        break
            if verbose:
                print("step size for A = %d", t)
            if not dosave:
                self._A = A
                return Ak_ls
            else:
                self._A = Ak_ls

    def WienerFilter(self, output_stereo=True):
        """

        Parameters
        ----------
        output_stereo

        Returns
        -------

        """
        A = self._A
        W = self._W
        H = self._H
        sigmab = self._sigmab

        F, N, I = self._XX.shape[:3]  # (F x N x I x I)
        J = A.shape[1]  # A (I x J)

        sigmax = np.zeros((F, N, I, I), dtype=np.complex)

        V = np.zeros((F, N, J))
        for j in range(J):
            Wj = W[:, self._comp_dict[j]]
            Hj = H[self._comp_dict[j], :]
            Vj = Wj.dot(Hj)
            V[:, :, j] = Vj

        sigmax[:, :, 0, 0] = sigmab
        sigmax[:, :, 1, 1] = sigmab

        for j in range(J):
            sigmax[:, :, 0, 0] += (np.abs(A[0, j]) ** 2) * V[:, :, j]
            sigmax[:, :, 0, 1] += (A[0, j] * A[1, j].conj()) * V[:, :, j]
            sigmax[:, :, 1, 1] += (np.abs(A[1, j]) ** 2) * V[:, :, j]
        sigmax[:, :, 1, 0] = sigmax[:, :, 0, 1].conj()

        # compute the inverse of sigmax
        inv_sigmax = np.zeros((F, N, I, I), dtype=np.complex)
        log_detx = sigmax[:, :, 0, 0] * sigmax[:, :, 1, 1] - np.abs(sigmax[:, :, 0, 1]) ** 2

        inv_sigmax[:, :, 0, 0] = sigmax[:, :, 1, 1] / log_detx
        inv_sigmax[:, :, 0, 1] = -sigmax[:, :, 0, 1] / log_detx
        inv_sigmax[:, :, 1, 1] = sigmax[:, :, 0, 0] / log_detx
        inv_sigmax[:, :, 1, 0] = inv_sigmax[:, :, 0, 1].conj()

        mat_tmp = np.zeros((F, N, I), dtype=np.complex)
        mat_tmp[:, :, 0] = inv_sigmax[:, :, 0, 0] * self._X[:, :, 0] + \
                           inv_sigmax[:, :, 0, 1] * self._X[:, :, 1]
        mat_tmp[:, :, 1] = inv_sigmax[:, :, 1, 0] * self._X[:, :, 0] + \
                           inv_sigmax[:, :, 1, 1] * self._X[:, :, 1]

        if output_stereo:
            self._sources = np.zeros((F, N, I, J), dtype=np.complex)
        else:
            self._sources = np.zeros((F, N, J), dtype=np.complex)

        for j in range(J):
            sources_j_mono = V[:, :, j]*(mat_tmp[:, :, 0]*A[0, j].conj() +
                                         mat_tmp[:, :, 1]*A[1, j].conj())
            if output_stereo:
                self._sources[:, :, 0, j] = sources_j_mono*A[0, j]
                self._sources[:, :, 1, j] = sources_j_mono*A[1, j]
            else:
                self._sources[:, :, j] = sources_j_mono
