# TODO: test everything.
# TODO: Determine whento use svd and when to use reduce_rank_svd

import numpy as np
import scipy.linalg
import sys

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
            # tuple is (M1, M2, M3, L)
            self.length = input[3]
            self.mps = [input[0]]
            for i in range(0, self.length-2):
                self.mps.append(input[1])
            self.mps.append(input[2])
        else:
            self.mps = input
            self.length = len(self.mps)
        self.chi = chi
        assert self.length > 2, "Length cannot be shorter than 3."
        assert self.length < 46, "MPS cannot right normalize if length is > 45"

    def __str__(self):
        print(self.mps)
        return ''

    def get(self):
        return(self.mps)

    def __getitem__(self, index):
        return(self.mps[index])

    def __setitem__(self, index, value):
        self.mps[index] = value

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
        l = self.length-1
        right = self.mps[l]
        u,s,v = scipy.linalg.svd(right, full_matrices=False, lapack_driver='gesvd')
        s = np.diag(s)

        self.mps[l] = v
        self.mps[l-1] = np.einsum('uxl, Rx -> uRl', self.mps[l-1], u)
        self.mps[l-1] = np.einsum('uxl, Rx -> uRl', self.mps[l-1], s)
        for i in range(l-1, 1, -1):
            right = self.mps[i] # Get tensor
            sh = right.shape
            right = np.reshape(right, [sh[0]*sh[1], sh[2]]) # reshape u and r into one dimension for svd
            # Now indexed M((U,R),L)
            right = np.einsum('ab->ba', right) # now indexed M(L,(U,R))
            u, s, v = scipy.linalg.svd(right, full_matrices=False, lapack_driver='gesvd')
            s = np.diag(s)

            # reshape v back into a rank-3 tensor and replace it
            right = np.einsum('ab->ba', right)  # now indexed M((U,R), L)
            v = np.reshape(v, [sh[0], sh[1], sh[2]])  # now U,R,L
            self.mps[i] = v

            self.mps[i-1] = np.einsum('uxl, Rx -> uRl', self.mps[i-1], u)
            self.mps[i-1] = np.einsum('uxl, Rx -> uRl', self.mps[i-1], s)

        right = self.mps[1]
        sh = right.shape
        right = np.reshape(right, [sh[0]*sh[1], sh[2]]) # reshape u and r into one dimension for svd
        right = np.einsum('ab->ba', right) # now indexed M(L,(U,R))
        u, s, v = scipy.linalg.svd(right, full_matrices=False, lapack_driver='gesvd')
        s = np.diag(s)

        # reshape u back into a rank-3 tensor and replace it
        right = np.einsum('ab->ba', right) # now indexed M((U,R), L)
        v = np.reshape(v, [sh[0], sh[1], sh[2]]) # Now U, R, L
        self.mps[1] = v
        self.mps[0] = np.einsum('ux, Rx -> uR', self.mps[0], u)
        self.mps[0] = np.einsum('ux, Rx -> uR', self.mps[0], s)

    def leftNormalizeIndex(self, index):
        '''
        Left normalize the MPS at index, multiplying s and v into the previous
        tensor.
        '''
        assert (index >= 0 and index < self.length-1), "Index Invalid"

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
        assert (index > 0 and index <= self.length-1), "Index Invalid"
        l = self.length-1
        if (index == l):
            right = self.mps[index]
            u, s, v = scipy.linalg.svd(right, full_matrices=False, lapack_driver='gesvd')
            s = np.diag(s)
            self.mps[l] = v
            self.mps[l-1] = np.einsum('uxl, Rx -> uRl', self.mps[l-1], u)
            self.mps[l-1] = np.einsum('uxl, Rx -> uRl', self.mps[l-1], s)
            return

        right = self.mps[index]
        sh = right.shape
        right = np.reshape(right, [sh[0]*sh[1], sh[2]]) # reshape u and r into one dimension for svd
        right = np.einsum('ab->ba', right) # now indexed M(L,(U,R))
        u, s, v = scipy.linalg.svd(right, full_matrices=False, lapack_driver='gesvd')
        s = np.diag(s)

        right = np.einsum('ab->ba', right)
        v = np.reshape(v, [sh[0], sh[1], sh[2]])  # now U,R,L
        self.mps[index] = v

        if (index == 1):
            self.mps[index-1] = np.einsum('ux, Rx -> uR', self.mps[index-1], u)
            self.mps[index-1] = np.einsum('ux, Rx -> uR', self.mps[index-1], s)
        else:
            self.mps[index-1] = np.einsum('uxl, Rx -> uRl', self.mps[index-1], u)
            self.mps[index-1] = np.einsum('uxl, Rx -> uRl', self.mps[index-1], s)
        return

    def contract(self, all=0, p=False):
        '''
        Contract without renormalization. All flag:
         = 0 finds <\psi|\psi>, assuming no normalization structure
         = -1 contracts up to the first tensor, should return 1 if MPS is correctly right normalized.
         = 1 takes the trace of the first tensor, which = <\psi|\psi> if the MPS is correctly right normalized.
        p = True prints the values as it contracts, False does not.
        '''
        # NOTE: Why does this work? I think next should be contracted with
        # xrl,xLR -> lLrR but what really works is xrl,xRL->lLrR.
        if all == 1:
            final = np.einsum('xr,xR->rR', self.mps[0], self.mps[0])
            final = np.einsum('rR,rR', final, final)
            assert np.isclose(np.imag(final), 0)
            return np.real(final)
        l = self.length
        edge = np.einsum('xl,xL->lL', self.mps[l-1], self.mps[l-1])
        next = np.einsum('xrl,xRL->lLrR', self.mps[l-2], self.mps[l-2]) #xLR->lLrR', self.mps[l-2], self.mps[l-2])
        final = np.einsum('rR,lLrR->lL', edge, next)
        if p: print(np.einsum('lL,lL', final, final))
        for i in range(l-2, 1, -1):
            next = np.einsum('xrl,xRL->lLrR', self.mps[i-1], self.mps[i-1]) #xLR->lLrR', self.mps[i-1], self.mps[i-1])
            final = np.einsum('rR,lLrR->lL', final, next)
            if p: print(np.einsum('lL,lL', final, final))
        if all == -1:
            final = np.einsum('lL,lL', final, final)
            if p: print(np.einsum('lL,lL', final, final))
            assert np.isclose(np.imag(final), 0)
            return np.real(final)
        if all == 0:
            final = np.einsum('lL,lL', final, edge)
            if p: print(np.einsum('lL,lL', final, final))
            assert np.isclose(np.imag(final), 0)
            return np.real(final)
        print("mps.contract() did not return anything!")
        sys.exit(0)



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
        last = self.state[l]
        # Initially, last is indexed up,right,down,left (last only has ul)
        # and operator is indexed up,right,down,left (no right at first)
        R = np.einsum('xl,UxL -> UlL', last, self.operator[l])
        # last is now indexed "in reverse" d l r
        R = np.einsum('xlL,xi -> lLi', R, last)
        # Now R is indexed bottomleft, middleleft, topleft = lLi
        self.Rdict[l-1] = R
        #print(np.einsum('abc,abc', R, R))
        for i in range(l-1, 0, -1):
            next = self.state[i]
            temp = np.einsum('xrl,URxL -> URrlL', next, self.operator[i])
            temp = np.einsum('URrlL,xij -> ijRrlL', temp, next)
            R = np.einsum('abc,icbalL -> lLi', R, temp)
            self.Rdict[i-1] = R
            #print(np.einsum('abc,abc', R, R))

    def findMatrix(self, index):
        '''
        Given an index, find the tensor to be used for the standard eigenvalue
        problem. Return it as a matrix (only two legs)
        '''

        # R is indexed bottom, middle, top (=lLi)
        if (index == 0):
            R = self.Rdict[index]
            matrix = np.einsum('lxi, UxD -> UilD', R, self.operator[index])
            sh = matrix.shape
            matrix = np.reshape(matrix, [sh[0]*sh[1], sh[2]*sh[3]])
            return matrix

        if (index == self.state.length-1):
            L = self.Ldict[index]
            matrix = np.einsum('rxj, UDx -> jUDr', L, self.operator[index])
            sh = matrix.shape
            matrix = np.reshape(matrix, [sh[0]*sh[1], sh[2]*sh[3]])
            return matrix
        # L is indexed bottom, middle, top (=rRj)
        L = self.Ldict[index]
        R = self.Rdict[index]

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
        return [m, v]

    def contract(self):
        '''
        Contract the entire network to find <\psi|O|\psi>
        '''
        self.R_Builder()
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
        self.R_Builder()

        for i in range(num):
            # Step 3: Right sweep
            tensor = self.findMatrix(0)
            eigen = self.davidson(tensor)
            sh = self.state[0].shape
            eigen[1] = np.reshape(eigen[1][0], [sh[0], sh[1]])
            self.state[0] = eigen[1]
            self.state.leftNormalizeIndex(0)
            L = self.state[0]
            Ltemp = np.einsum('xr,URx->URr', L, self.operator[0])
            L = np.einsum('xRr,xi->rRi', Ltemp, L) # Now indexed bottom, middle, top
            self.Ldict[1] = L

            for j in range(1, self.state.length-1):
                tensor = self.findMatrix(j)
                eigen = self.davidson(tensor)
                sh = self.state[j].shape
                eigen[1] = np.reshape(eigen[1][0], [sh[0], sh[1], sh[2]])
                self.state[j] = eigen[1]
                self.state.leftNormalizeIndex(j)
                Ltemp = np.einsum('xrl,URxL->URrlL', self.state[j], self.operator[j])
                Ltemp = np.einsum('xRrlL,xij->ijRrlL', Ltemp, self.state[j])
                L = np.einsum('ijRrlL,lLi->rRj', Ltemp, L)
                self.Ldict[j+1] = L
            # Step 4: Left sweep
            index = self.state.length-1
            tensor = self.findMatrix(index)
            eigen = self.davidson(tensor)
            sh = self.state[index].shape
            # NOTE: possible source of error here, since sh[0] and sh[1] may be flipped
            eigen[1] = np.reshape(eigen[1][0], [sh[0], sh[1]])
            self.state[index] = eigen[1]
            self.state.rightNormalizeIndex(index)
            last = self.state[index]
            R = np.einsum('xl,UxL -> UlL', last, self.operator[index])
            R = np.einsum('xlL,xi -> lLi', R, last) # Now R is indexed bottomleft, middleleft, topleft = lLi
            self.Rdict[index-1] = R

            for j in range(self.state.length-2, 0, -1):
                tensor = self.findMatrix(j)
                eigen = self.davidson(tensor)
                sh = self.state[j].shape
                eigen[1] = np.reshape(eigen[1][0], [sh[0], sh[1], sh[2]])
                self.state[j] = eigen[1]
                self.state.rightNormalizeIndex(j)
                Rtemp = np.einsum('xrl,URxL->URrlL', self.state[j], self.operator[j])
                Rtemp = np.einsum('xRrlL,xij->ijRrlL', Rtemp, self.state[j])
                R = np.einsum('ijRrlL,rRj->lLi', Rtemp, R)
                self.Rdict[j-1] = R

        return (self.state.get(), self.contract())

def buildIsingMPS(L=10, chi=10):
    '''
    Build an example MPS with the transfer matrix used to solve the ising model.
    '''
    J,k,H,T = 1,1,0,0.1
    K,h = J/(k*T), H/(k*T)
    V = np.array([[np.exp(K+h), np.exp(-K)], [np.exp(-K), np.exp(K-h)]])
    D, M = scipy.linalg.eig(V) # Eigenvalue decomposition of matrix
    D = np.diag(D)
    # Now V = M * D * M^-1
    G = M.dot(D**0.5).dot(scipy.linalg.inv(M)) # "Square root" of matrix
    # Because the transfer matrix is symmetric, corners are the same and so are edges and centers.
    cornerTensor = np.einsum('rx, xD -> rD', G, G)
    edgeTensor = np.einsum('ux, Rx, xj -> uRj', G, G, G)
    return MPS((cornerTensor, edgeTensor, cornerTensor, L), chi)

def buildExampleMPO(I=True, L=10, chi=10):
    '''
    Build an example MPO.
    '''
    #TODO: implement this function correctly
    if I:
        # They're all identity, but for the sake of clarity:
        V = np.array([[1, 0], [0, 1]])
        D, M = scipy.linalg.eig(V) # Eigenvalue decomposition of matrix
        D = np.diag(D)
        G = M.dot(D**0.5).dot(scipy.linalg.inv(M))
        edgeTensor = np.einsum('Ux, Rx, xD -> URD', G, G, G)
        middleTensor = np.einsum('Ux, Rx, xD, xL -> URDL', G, G, G, G)
        return MPS((edgeTensor, middleTensor, edgeTensor, L), chi)
    pass

def FullTest():
    # Generate an example MPS
    L,chi = 10,10
    testFlag = True
    print("Testing MPS initialization and full right normalization. ")
    mps = buildIsingMPS()
    print("Initialization of MPS successful. ")
    # Right normalize the example MPS
    #print(mps)
    mps.rightNormalize()
    print("Right normalization did not return any errors. ")

    # Contract the example MPS and assert it is 1
    c = mps.contract(-1)
    print("MPS contraction did not return any errors. ")
    if (np.isclose(c,2)):
        print("Tr(<\psi|\psi>) up to index 1 = 2, right normalization and contraction likely worked correctly.")
    else:
        print("Tr(<\psi|\psi>) up to index = " + str(c) + ", right normalization or contraction likely failed")
        testFlag = False

    print("\nTesting rightNormalizeIndex.")
    # Test rightNormalizeIndex by doing full right normalization one at a time
    mps = buildIsingMPS()
    for i in range(mps.length-1, 0, -1):
        mps.rightNormalizeIndex(i)
    print("rightNormalizeIndex did not return any errors at any index.")
    c1 = mps.contract(-1)
    print("MPS contraction did not return any errors. ")
    if (np.isclose(c1,2)):
        print("Tr(<\psi|\psi>) up to index 1 = 2, right normalization likely worked correctly.")
    else:
        print("Tr(<\psi|\psi>) up to index = " + str(c1) + ", right normalization likely failed")
        testFlag = False

    print("\nTesting leftNormalizeIndex.")
    mps.leftNormalizeIndex(0)
    print("Left normalization at index 0 did not return any errors.")
    c = mps.contract(1)
    if (np.isclose(c,2)):
        print("Tr(first tensor) = 2, left normalization at index 0 likely worked")
    else:
        print("Tr(first tensor) = " + str(c) + ", left normalization at index 0 failed.")
        testFlag = False

    # Make sure replacing and getting works.
    print("\nTesting replacing tensors as used later in Network class. ")
    mps = buildIsingMPS()
    mps2 = mps.get().copy()
    mps[1] = np.random.rand(2,2,2)
    print("__setitem__ did not return any errors.")
    if not np.isclose(np.real(mps2[1][0][0][0]), np.real(mps[1][0][0][0])):
        print("Replacement seems to have worked correctly. ")
    else:
        print("Replacement failed.")
        testFlag = False

    print("\nMPS class passed all tests.\n") if testFlag else print("\nMPS failed some tests.\n")

    testFlag = True
    mps = buildIsingMPS()
    print("\nTesting MPO initialization.")
    mpo = buildExampleMPO()
    print("Initialization of MPO did not return any errors.")
    print("Initializing Network class with example MPS and MPO.")
    network = Network(mps, mpo)

    print("\nTesting function R_Builder")
    network.state.rightNormalize()
    # Right normalize, then Tr(R) should be 2 for all
    network.R_Builder()
    # Is the MPO correctly built? Tr(R) halves each index
    print("R_Builder probably works based on tests I do not want to automate.")
    print("Example (identity) MPO likely built incorrectly.")

    print("\nTesting contraction with test MPO. ")
    c = network.contract()

    if (np.isclose(c, c1)):
        print("Contraction returns the same thing as MPS contraction. MPO was likely built correctly, contraction likely works.")
    else:
        print("MPS contraction != identity MPO contraction. MPO initialization or contraction worked incorrectly.")
        testFlag = False

    network = Network(mps, mpo)
    print("\nTesting findMatrix at index 0")

    network.state.rightNormalize()
    network.R_Builder()
    m = network.findMatrix(0)
    if (m.shape == (4,4,4)):
        print("m has correct shape at index 0.")
    print("The rest of findMatrix cannot be tested until solve() is tested. ")


    print("\nTesting solve() with brand new network.")
    mpo = buildExampleMPO()
    network = Network(mps, mpo)
    mps = buildIsingMPS()
    a = network.solve(5)
    print("Network solve() did not return any errors, no idea if correct. ")

    print("\nNetwork class passed all tests.") if testFlag else print("\nNetwork class failed some tests.")

    return

FullTest()
