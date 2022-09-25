#from math import comb
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

from .arap_utils import *

import torch
import torch.nn

class ARAP(torch.nn.Module):
    def __init__(self, pnts: torch.Tensor, tris: np.ndarray, handles, laplacian='cotangent', iA=None):
        """
            laplacian: 'cotangent' or 'combinatorial'
        """
        super().__init__()

        self.device = pnts.device

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
        
        # cache everything that is needed
        self.L = L
        
        # form and invert the system
        if iA is None:
            A = scipy.sparse.bmat([
                [L.T@L, C.T],
                [C, szeros((Nc, Nc))]
            ]) # shape: [Np+Nc, Np+Nc]
            iA = np.linalg.pinv(A.toarray()).astype(np.float32)
        
        # extract neighbors and weights, removing entries on the diagonal
        # these summed and negated should equal the diagonal for the Laplacian

        Lnbrs, Lwgts = matrix_indices_and_weights(L, zero_diag=False)
        Ldiag = -np.sum(Lwgts, axis=-1)

        self.P = pnts.shape[0]
        self.D = pnts.shape[1]
        self.C = len(anchors)
        self.anchors = anchors
        self.pnts = torch.nn.Parameter( pnts, requires_grad=False )
        self.Lnbrs = torch.nn.Parameter( torch.from_numpy(Lnbrs.astype(np.int64) ).to(self.device), requires_grad=False )
        self.Lwgts = torch.nn.Parameter( torch.from_numpy(Lwgts.astype(np.float32) ).to(self.device), requires_grad=False )
        self.Ldiag = torch.nn.Parameter( torch.from_numpy(Ldiag.astype(np.float32)).to(self.device), requires_grad=False )
        self.iA = torch.nn.Parameter( torch.from_numpy(iA).to(self.device), requires_grad=False )
        
        return iA

    def _mul_L(self, pnts):
        return torch.sum( pnts[self.Lnbrs, :]*self.Lwgts[...,None], dim=1) + self.Ldiag[...,None]*pnts

    def _estimate_rhs(self, old_pnts, new_pnts):
        # get the neighborhoods
        O = -( old_pnts[self.Lnbrs, :] - old_pnts[:,None,:])*self.Lwgts[:,:,None]
        N = -( new_pnts[self.Lnbrs, :] - new_pnts[:,None,:])*self.Lwgts[:,:,None]
        
        # compute the rotation between the neighborhoods
        cov = N.permute(0,2,1) @ O
        U, s, Vt = torch.linalg.svd(cov)

        # fix up the jacobian, if needed
        ones = torch.ones_like(s[:, 0], device=self.device)
        sign = torch.sign(torch.det(U @ Vt))
        s_new = torch.stack((ones,ones,sign), dim=-1)
        R = U @ (s_new[...,None]*Vt)

        # use average rotation for the edge to rotate the edge vectors
        d_new = torch.sum( ((R[self.Lnbrs, ...] + R[:,None,:,:]) @ O[..., None])[...,0]*(0.5*self.Lwgts[:,:,None]), dim=1 )
        return self._mul_L(d_new)

    def forward(self, num_iter=1):
        crhs = torch.stack([self.handles[a] for a in self.anchors], dim=0) # constraints
        lrhs = self._mul_L(self._mul_L(self.pnts))
        rhs = torch.cat([lrhs, crhs], dim=0)
        new_pnts = (self.iA @ rhs)[:self.P]
        for iter in range(num_iter):
            new_rhs = self._estimate_rhs(self.pnts, new_pnts)
            rhs = torch.cat([new_rhs, crhs], dim=0)
            new_pnts = (self.iA @ rhs)[:self.P]
        return new_pnts


