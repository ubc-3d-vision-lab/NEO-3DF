#from math import comb
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .arap_utils import *


class ARAP:
    def __init__(self, pnts, tris, handles, laplacian='cotangent'):
        """
            laplacian: 'cotangent' or 'combinatorial'
        """

        anchors = list(handles.keys())
        self.handles = handles

        # number of points and anchors
        Np = len(pnts)
        Nc = len(anchors)

        if laplacian == 'cotangent':
            L = cotangent_laplacian(pnts=pnts, tris=tris)
        else:
            L = combinatorial_laplacian(pnts=pnts, tris=tris)

        # build the constraint matrix
        vals = np.ones(len(anchors))
        rows = np.arange(len(anchors))
        cols = np.array(anchors)
        C = scipy.sparse.csr_matrix( (vals, (rows, cols)), shape=(Nc, Np) )

        # form and invert the system
        A = scipy.sparse.bmat([
            [L.T@L, C.T],
            [C, szeros((Nc, Nc))]
        ]) # shape: [Np+Nc, Np+Nc]

        # cache everything that is needed
        self.L = L
        self._iA = scipy.sparse.linalg.factorized(A)

        # extract neighbors and weights, removing entries on the diagonal
        # these summed and negated should equal the diagonal for the Laplacian
        self.anchors = anchors
        self.pnts = pnts.copy()

        self.Lnbrs, self.Lwgts = matrix_indices_and_weights(L, zero_diag=False)
        self.Ldiag = -np.sum(self.Lwgts, axis=-1)

    def _iA(self, rhs):
        return self.iA @ rhs

    def _mul_L(self, pnts):
        return np.sum(pnts[self.Lnbrs, :]*self.Lwgts[...,None], axis=1) + self.Ldiag[...,None]*pnts

    def _estimate_rhs(self, old_pnts, new_pnts):
        # get the neighborhoods
        O = -( old_pnts[self.Lnbrs, :] - old_pnts[:,None,:])*self.Lwgts[:,:,None]
        N = -( new_pnts[self.Lnbrs, :] - new_pnts[:,None,:])*self.Lwgts[:,:,None]

        # compute the rotation between the neighborhoods
        cov = N.transpose(0,2,1) @ O  # shape [np, 3, 3]
        U, s, Vt = np.linalg.svd(cov)

        # fix up the jacobian, if needed
        ones = np.ones_like(s[:, 0])
        sign = np.sign(np.linalg.det(U @ Vt))
        s_new = np.stack((ones,ones,sign), axis=-1)
        R = U @ (s_new[...,None]*Vt)

        # use average rotation for the edge to rotate the edge vectors
        d_new = np.sum( ((R[self.Lnbrs, ...] + R[:,None,:,:]) @ O[..., None])[...,0]*(0.5*self.Lwgts[:,:,None]), axis=1 )
        return self._mul_L(d_new)

    def __call__(self, num_iter=1):
        crhs = np.row_stack([self.handles[a] for a in self.anchors]) # constraints
        lrhs = self._mul_L(self._mul_L(self.pnts))
        rhs = np.row_stack((lrhs, crhs))
        new_pnts = self._iA(rhs)[:len(self.pnts)]
        for iter in range(num_iter):
            new_rhs = self._estimate_rhs(self.pnts, new_pnts)
            rhs = np.row_stack((new_rhs, crhs))
            new_pnts = self._iA(rhs)[:len(self.pnts)]
        return new_pnts



