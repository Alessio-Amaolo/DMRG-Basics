# DMRG code to find the ground state of quantum systems using a variational method
# Based on the methods in chapter 6 of this paper: https://arxiv.org/pdf/1008.3477.pdf

import numpy as np

class MPS():
    '''
    Represents an Matrix Product State, ndarray with additional functions.
    Takes a list of the MPS, or alternatively a first tensor, last tensor,
    the middle tensor and the length as a tuple.

    Input should be an ndarray with the format [tensor_1, tensor_2, ...]
    where tensor_n is assumed to be a rank-3 tensor as a 3d ndarray.

    Tensors are indexed up, right, left. Hence the shape of this MPS is
    (L, n1, n2, n3) where L is the length and n is the dimension of each tensor,
    with n1=n2=n3 but n1=up, n2=right, n3=left.
    '''
    def __init__(self, input, chi):
        if isinstance(input, tuple):
            # TODO: implement passing (M1, M2, M3, L)
            pass
        else:
            self.mps = input
        self.chi = chi

    def get(self):
        return(self.mps[0])

    def reduce_rank_svd(self, mat):
        '''
        SVD then reduce rank. Return u, s, and v matrices.
        Second argument selects the number of columns to keep
        '''
        u, s, v = scipy.linalg.svd(mat, full_matrices=False, lapack_driver='gesvd')
        u = u[:,:self.chi]
        s = np.diag(s[:self.chi])
        v = v[:self.chi,:]
        return(u, s, v)

    def rightNormalize(self):
        '''
        Right normalize the last tensor by SVD, and put u in as the tensor.
        Multiply s and v into the next tensor. Repeat to right normalize
        the entire MPS.
        '''
        for i in range(self.mps.shape[0]-1, 1, -1):
            right = self.mps[i] # Get tensor
            sh = right.shape
            right = np.reshape(right, [sh[0]*sh[1], sh[2]]) # reshape u and r into one dimension for svd
            u, s, v = self.reduce_rank_svd(right)

            # reshape u back into a rank-3 tensor and replace it
            newShape = self.chi if u.shape[1] > self.chi else u.shape[1]
            u = np.reshape(u, [sh[0], sh[1], newShape])
            self.mps[i] = u

            self.mps[i-1] = np.einsum('uxl, Rx -> uRl', self.mps[i-1], v)
            self.mps[i-1] = np.einsum('uxl, Rx -> uRl', self.mps[i-1], s)

    def leftNormalize(self, index):
        '''
        Left normalize the MPS starting at index and working to the right.
        '''
        for i in range(index, self.mps.shape[0]-1):
            left = self.mps[i]
            sh = right.shape
            left = np.einsum('url->ulr', left)
            left = np.reshape(left, [sh[0]*sh[1], sh[2]]) # reshape u and r into one dimension for svd
            u, s, v = self.reduce_rank_svd(left) # regular svd?

            # reshape u back into a rank-3 tensor and replace it
            newShape = self.chi if u.shape[1] > self.chi else u.shape[1]
            u = np.reshape(u, [sh[0], sh[1], newShape])
            u = np.einsum('ulr->url', u)
            self.mps[i] = u

            self.mps[i+1] = np.einsum('xL,urx -> urL', v, self.mps[i+1])
            self.mps[i+1] = np.einsum('xL,urx -> urL', s, self.mps[i+1])

class MPO(MPS):
    pass

class Solver():
    '''
    Variational ground state solver with initial guess of state and given
    operator in Matrix Product Operator (MPO) form.
    '''
    def __init__(self, initialMPS, mpo):
        self.state = initialMPS
        self.operator = mpo
        self.Rdict = {}
        self.Ldict = {}

    def R_Builder(self):
        '''
        Build all R for a single variational cycle. Given an MPS and MPO, stores
        all possible in a dictionary with the keys as the index the R goes to,
        from 0 to L-2 (if the last tensor is empty, there is no R).
        '''
        l = self.state.shape[0]
        last = self.state[l-1]
        R = np.einsum('xrl,URxL -> URrlL', last, self.operator[l-1])
        # last is now indexed "in reverse" d l r
        # now R is indexed leftTop, rightTop, rightMiddle, rightBottom, leftBottom, leftMiddle
        R = np.einsum('URrlL,xij -> ijRrlL', R, last)
        self.Rdict[l-1] = R # might need to copy R, as it might put a pointer to R there
        for i in range(l-2, 0, -1):
            next = self.state[i]
            temp = np.einsum('xrl,URxL -> URrlL', last, self.operator[i])
            temp = np.einsum('URrlL,xij -> ijRrlL', temp, last)
            R = np.einsum('xjRrlL,ixLlab -> ijRrab', R, temp)
            self.Rdict[i] = R

    def L_Builder(self):
        '''
        Build all L for a single variational cycle.
        '''
        # TODO: Do this, same as R.
        pass

    def findMatrix(self, index):
        '''
        Given an index, find the tensor to be used for the standard eigenvalue
        problem.
        '''
        R = self.Rdict[index]
        L = self.Ldict[index]
        # TODO: Multiply them together with an extra state + O, return the tensor
        pass

    def standardEigen(self, tensor):
        '''
        Solve tensor1*tensor2 - \lambda * tensor2 = 0 for the smallest
        eigenvalue/eigentensor pair.
        '''
        # TODO: Do this.
        pass

    def contract(self):
        '''
        Contract the entire network to find <\psi|O|\psi>
        '''
        pass

    def solve(self, num):
        '''
        Find the minimum of <\psi|O|\psi> for operator O.
        Returns (wavefunction, energy) tuple
        '''
        for i in range(num):
            # Right sweep
            for j in range(0, self.state.shape[0]-1):
                tensor = findMatrix(j)
                self.state[j] = standardEigen(tensor)

            # Left sweep
            for j in range(self.state.shape[0]-1, 0, -1):
                tensor = findMatrix(j)
                self.state[j] = standardEigen(tensor)

        return (self.state, contract())
