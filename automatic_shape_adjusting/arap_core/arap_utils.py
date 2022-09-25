from matplotlib.pyplot import axis
import numpy as np
import scipy.sparse as sp

def szeros(shape):
    return sp.csr_matrix(shape)

def triangle_vertices(pnts, tris):
    return pnts[tris[:,0],:], pnts[tris[:,1],:], pnts[tris[:,0],:]

def triangle_normal_area_angles(pnts, tris):
    eps = 1e-7

    # get vertex coordinates
    a,b,c = triangle_vertices(pnts, tris)

    # edge vectors
    ab, bc, ca = b-a, c-b, a-c
    lab, lbc, lca = [ np.maximum(np.linalg.norm(v, axis=1), eps) for v in (ab, bc, ca)]

    # get the normal
    normal = -np.cross(ca, ab, axis=1)
    area = np.maximum(np.linalg.norm(normal, axis=1), eps)

    # compute interior angles
    angles = np.column_stack((
        np.arcsin( area / (lca * lab) ),
        np.arcsin( area / (lab * lbc) ),
        np.arcsin( area / (lbc * lca) )
    ))

    return normal / area[:, None], area / 2.0, angles


def combinatorial_laplacian(pnts, tris):
    a,b,c = tris[:,0], tris[:,1], tris[:,2]
    v = 0.5 * np.ones(len(a))
    rows, cols, vals = [], [], []

    # edge b--c
    rows += [b, b, c, c]
    cols += [b, c, c, b]
    vals += [v, -v, v, -v]

    # edge c--a
    rows += [c, c, a, a]
    cols += [c, a, a, c]
    vals += [v, -v,  v, -v]

    # edge a--b    
    rows += [a, a, b, b]
    cols += [a, b, b, a]
    vals += [ v, -v,  v, -v]

    L = sp.csr_matrix( (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols)) ), shape=(len(pnts), len(pnts)) )

    return L


def cotangent_laplacian(pnts, tris, angles=None):
    if angles is None:
        angles = triangle_normal_area_angles(pnts, tris)[-1]

    a,b,c = tris[:,0], tris[:,1], tris[:,2]
    ca,cb,cc = 0.5*(1.0/np.tan(np.minimum(angles, np.pi/2))).T
    rows, cols, vals = [], [], []

    # edge b--c
    rows += [b, b, c, c]
    cols += [b, c, c, b]
    vals += [ca, -ca, ca, -ca]

    # edge c--a
    rows += [c, c, a, a]
    cols += [c, a, a, c]
    vals += [cb, -cb, cb, -cb]

    # edge a--b    
    rows += [a, a, b, b]
    cols += [a, b, b, a]
    vals += [cc, -cc, cc, -cc]

    L = sp.csr_matrix( (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols)) ), shape=(len(pnts), len(pnts)) )

    return L



def matrix_indices_and_weights(M: sp.csr_matrix, zero_diag: bool=True):
    """
    returns dense array of column indices and weights from a csr_matrix optionally zeroing diagonal coefficients
    """
    max_nbrs = np.max(M.indptr[1:] - M.indptr[0:-1])
    nbrs = np.repeat(np.arange(M.shape[0])[:, None], max_nbrs, axis=1)
    wgts = np.zeros((M.shape[0], max_nbrs))
    for i in range(nbrs.shape[0]):
        s, e = M.indptr[i], M.indptr[i+1]
        cnt = e - s
        nbrs[i, :cnt] = M.indices[s:e]
        wgts[i, :cnt] = M.data[s:e]
    if zero_diag:
        wgts[nbrs == np.arange(M.shape[0])[:, None]] = 0.0
    return nbrs, wgts






def test_case():
    import matplotlib.pyplot as plt

    pnts = np.array((
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0)
    ))

    tris = np.array((
        (0, 1, 2),
        (0, 2, 3)
    ))

    L_cob = np.zeros((len(pnts), len(pnts))) +  combinatorial_laplacian(pnts, tris)
    print(L_cob)
    plt.imshow(L_cob)
    plt.show()

    L_cot = np.zeros((len(pnts), len(pnts))) + cotangent_laplacian(pnts, tris)
    print(L_cot)
    plt.imshow(L_cot)
    plt.show()



if __name__ == '__main__':
    test_case()

