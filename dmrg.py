# DMRG code to find the ground state of quantum systems using a variational method
# Based on the methods in chapter 6 of this paper: https://arxiv.org/pdf/1008.3477.pdf

# TODO: test everything.
# TODO: Determine whento use svd and when to use reduce_rank_svd

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
        self.length = self.mps.shape[0]

    def get(self):
        return(self.mps)

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
        l = self.mps.shape[0]-1
        right = self.mps[l]
        u, s, v = self.reduce_rank_svd(right)
        self.mps[l] = u
        self.mps[l-1] = np.einsum('uxl, Rx -> uRl', self.mps[l-2], v)
        self.mps[l-1] = np.einsum('uxl, Rx -> uRl', self.mps[l-2], s)

        for i in range(self.mps.shape[0]-2, 2, -1):
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

        right = self.mps[1]
        sh = right.shape
        right = np.reshape(right, [sh[0]*sh[1], sh[2]]) # reshape u and r into one dimension for svd
        u, s, v = self.reduce_rank_svd(right)

        # reshape u back into a rank-3 tensor and replace it
        newShape = self.chi if u.shape[1] > self.chi else u.shape[1]
        u = np.reshape(u, [sh[0], sh[1], newShape])
        self.mps[1] = u

        self.mps[0] = np.einsum('ux, Rx -> uR', self.mps[0], v)
        self.mps[0] = np.einsum('ux, Rx -> uR', self.mps[0], s)

    def leftNormalizeIndex(self, index):
        '''
        Left normalize the MPS at index.
        '''
        assert (index >= 0 and index <= self.length-1), "Index Invalid"

        if (index == 0):
            left = self.mps[0]
            u, s, v = self.reduce_rank_svd(left)
            self.mps[0] = u
            self.mps[1] = np.einsum('xL,urx -> urL', v, self.mps[1])
            self.mps[1] = np.einsum('xL,urx -> urL', s, self.mps[1])
            return

        # TODO: fix this so it doens't iterate over 1 number.
        for i in range(index, index): # self.mps.shape[0]-1):
            left = self.mps[i]
            sh = left.shape
            left = np.einsum('url->ulr', left)
            left = np.reshape(left, [sh[0]*sh[1], sh[2]]) # reshape u and r into one dimension for svd
            u, s, v = self.reduce_rank_svd(left) # regular svd?

            # reshape u back into a rank-3 tensor and replace it
            newShape = self.chi if u.shape[1] > self.chi else u.shape[1]
            u = np.reshape(u, [sh[0], sh[1], newShape])
            u = np.einsum('ulr->url', u)
            self.mps[i] = u

            if (i == self.length-2):
                self.mps[i+1] = np.einsum('xL,ux -> uL', v, self.mps[i+1])
                self.mps[i+1] = np.einsum('xL,ux -> uL', s, self.mps[i+1])
            else:
                self.mps[i+1] = np.einsum('xL,urx -> urL', v, self.mps[i+1])
                self.mps[i+1] = np.einsum('xL,urx -> urL', s, self.mps[i+1])
            return

    def rightNormalizeIndex(self, index):
        assert (index >= 0 and index <= self.length-1), "Index Invalid"
        if (index == self.length-1):
            right = self.mps[index]
            u, s, v = self.reduce_rank_svd(right)
            self.mps[index] = u
            self.mps[index-1] = np.einsum('Rx,uxl -> uRl', v, self.mps[index-1])
            self.mps[index-1] = np.einsum('Rx,uxl -> uRl', s, self.mps[index-1])
            return

        right = self.mps[index]
        sh = right.shape
        right = np.reshape(right, [sh[0]*sh[1], sh[2]]) # reshape u and r into one dimension for svd
        u, s, v = self.reduce_rank_svd(left) # regular svd?

        # reshape u back into a rank-3 tensor and replace it
        newShape = self.chi if u.shape[1] > self.chi else u.shape[1]
        u = np.reshape(u, [sh[0], sh[1], newShape])
        self.mps[index] = u

        if (index == 1):
            self.mps[index-1] = np.einsum('rx,ux -> ur', v, self.mps[index-1])
            self.mps[index-1] = np.einsum('rx,ux -> ur', s, self.mps[index-1])
        else:
            self.mps[index-1] = np.einsum('Rx,uxl -> uRl', v, self.mps[index-1])
            self.mps[index-1] = np.einsum('Rx,uxl -> uRl', s, self.mps[index-1])
        return


# class MPO(MPS):
#     def rightNormalize(self):
#         pass
#     def leftNormalize(self):
#         pass

class Network():
    '''
    Variational ground state solver with initial guess of state (MPS) and given
    operator in Matrix Product Operator (MPO) form.
    '''
    def __init__(self, initialMPS, mpo):
        self.state = initialMPS
        self.operator = mpo
        self.Rdict = {}
        self.Ldict = {}
        self.storeShape = None

    def R_Builder(self):
        '''
        Build all R for a single variational cycle. Given an MPS and MPO, stores
        all possible in a dictionary with the keys as the index the R goes to,
        from 0 to L-2 (if the last tensor is empty, there is no R).
        '''
        l = self.state.length-1
        last = self.state.get()[l]
        # Initially, last is indexed up,right,down,left (last only has ul)
        # and operator is indexed up,right,down,left (no right at first)
        R = np.einsum('xl,UxL -> UlL', last, self.operator[l])
        # last is now indexed "in reverse" d l r
        R = np.einsum('xlL,xi -> lLi', R, last)
        # Now R is indexed bottomleft, middleleft, topleft = lLi
        # might need to copy R, as it might put a pointer to R there
        self.Rdict[l-1] = R
        for i in range(l-1, 1, -1):
            next = self.state.get()[i]
            temp = np.einsum('xrl,URxL -> URrlL', next, self.operator[i])
            temp = np.einsum('URrlL,xij -> ijRrlL', temp, next)
            R = np.einsum('abc,icbalL -> lLi', R, temp)
            self.Rdict[i-1] = R

    def findMatrix(self, index):
        '''
        Given an index, find the tensor to be used for the standard eigenvalue
        problem. Return it as a matrix (only two legs)
        '''

        R = self.Rdict[index]
        # R is indexed bottom, middle, top (=lLi)
        if (index == 0):
            matrix = np.einsum('lxi, UxD -> UilD', R, self.operator[index])
            sh = matrix.shape
            matrix = np.reshape(matrix, [sh[0]*sh[1], sh[2]*sh[3]])
            return matrix

        # L is indexed bottom, middle, top (=rRj)
        L = self.Ldict[index]
        if (index == self.state.length-1):
            matrix = np.einsum('rxj, UDx -> jUDr', L, self.operator[index])
            sh = matrix.shape
            matrix = np.reshape(matrix, [sh[0]*sh[1], sh[2]*sh[3]])
            return matrix

        matrix = np.einsum('lxi, UxDL -> UilDL', R, self.operator[index])
        matrix = np.einsum('UilDx, rxj -> UilDrj', matrix, L)
        matrix = np.einsum('UilDrj -> jUilDr', matrix)
        sh = matrix.shape
        self.storeShape = sh
        matrix = np.reshape(matrix, [sh[0]*sh[1]*sh[2], sh[3]*sh[4]*sh[5]])
        return matrix

    def davidson(self, tensor):
        '''
        Solve tensor1*tensor2 - \lambda * tensor2 = 0 for the smallest
        eigenvalue/eigentensor pair.

        Actually: Solve matrix*vector - \lambda * vector = 0 for the smallest
        eigenvalue/eigenvector pair using Davidson's method.
        '''
        # TODO: Actually implement Davidson's Method.
        e, v = np.linalg.eig(tensor)
        m = np.amin(e)
        v = v[np.where(e == m)]
        return (m, v)

    def contract(self):
        '''
        Contract the entire network to find <\psi|O|\psi>
        '''
        R_Builder()
        R = self.Rdict[0]
        next = self.state.get()[0]
        temp = np.einsum('xr,URx -> URr', next, self.operator[0])
        temp = np.einsum('xRr,xj -> jRr', temp, next)
        R = np.einsum('jRr,jRr', R, temp)
        return R

    def solve(self, num):
        '''
        Find the minimum of <\psi|O|\psi> for operator O.
        Returns (wavefunction, energy) tuple
        Argument num: number of cycles to compute.
        '''
        # TODO: Implement convergence or cycle until En-E(n-1) < \epsilon

        # Step 1: Start from an initial guess that is right normalized
        self.state.rightNormalize()

        # Step 2: Calculate the R expressions iteratively
        R_Builder()

        for i in range(num):
            # Step 3: Right sweep
            tensor = findMatrix(0)
            eigen = davidson(tensor)
            sh = self.state.get()[j].shape
            eigen[1] = np.reshape(eigen[1], [sh[0], sh[1]])
            # TODO: Probably doesn't work, implement changing state method
            self.state.get()[0] = eigen[1]
            self.state.leftNormalizeIndex(0)
            L = self.state.get()[0]
            L = np.einsum('xr,URx->URr', L, self.operator[0])
            L = np.einsum('xRr,xi->rRi') # Now indexed bottom, middle, top
            self.Ldict[1] = L

            for j in range(1, self.state.length-2):
                tensor = findMatrix(j)
                eigen = davidson(tensor)
                sh = self.state.get()[j].shape
                eigen[1] = np.reshape(eigen[1], [sh[0], sh[1], sh[2]])
                self.state.get()[j] = eigen[1]
                self.state.leftNormalizeIndex(j)
                Ltemp = np.einsum('xrl,URxL->URrlL', self.state.get()[j], self.operator[i])
                Ltemp = np.einsum('xRrlL,xij->ijRrlL', Ltemp, self.state.get()[j])
                L = np.einsum('ijRrlL,lLi->rRj', Ltemp, L)
                self.Ldict[j+1] = L


            # Step 4: Left sweep
            index = self.state.length-1
            tensor = findMatrix(index)
            eigen = davidson(tensor)
            sh = self.state.get()[index].shape
            # NOTE: possible source of error here, since sh[0] and sh[1] may be flipped
            eigen[1] = np.reshape(eigen[1], [sh[0], sh[1]])
            self.state.get()[index] = eigen[1]
            self.state.rightNormalizeIndex(index)

            last = self.state.get()[index]
            R = np.einsum('xl,UxL -> UlL', last, self.operator[index])
            R = np.einsum('xlL,xi -> lLi', R, last)
            # Now R is indexed bottomleft, middleleft, topleft = lLi
            self.Rdict[index-1] = R
            # might need to copy R, as it might put a pointer to R there

            for j in range(self.state.length-2, 1, -1):
                tensor = findMatrix(j)
                eigen = davidson(tensor)
                sh = self.state[j].shape
                eigen[1] = np.reshape(eigen[1], [sh[0], sh[1], sh[2]])
                self.state[j] = eigen[1]
                self.state.rightNormalizeIndex(j)

                Rtemp = np.einsum('xrl,URxL->URrlL', self.state[j], self.operator[i])
                Rtemp = np.einsum('xRrlL,xij->ijRrlL', Rtemp, self.state[j])
                R = np.einsum('ijRrlL,rRj->lLi', Rtemp, R)
                self.Rdict[j-1] = R

            # Step 5: Repeat sweeps until done

        return (self.state, self.contract())
