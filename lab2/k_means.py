######################### DISCLAIMER #####################
# I did not write this code, I snagged it from Stack Exchange
# from user "denis".

# kmeans.py using any of the 20-odd metrics in scipy.spatial.distance
# kmeanssample 2 pass, first sample sqrt(N)

from __future__ import division
import random
import numpy as np
from scipy.spatial.distance import cdist  # $scipy/spatial/distance.py
    # http://docs.scipy.org/doc/scipy/reference/spatial.html
from scipy.sparse import issparse  # $scipy/sparse/csr.py

__date__ = "2011-11-17 Nov denis"
    # X sparse, any cdist metric: real app ?
    # centers get dense rapidly, metrics in high dim hit distance whiteout
    # vs unsupervised / semi-supervised svm

#...............................................................................
def kmeans( X, centers, delta=.001, maxiter=10, metric="euclidean", p=2, verbose=1 ):
    """ centers, clusters, distances = kmeans( X, initial centers ... )
    in:
        X N x dim  may be sparse
        centers k x dim: initial centers, e.g. random.sample( X, k )
        delta: relative error, iterate until the average distance to centers
            is within delta of the previous average distance
        maxiter
        metric: any of the 20-odd in scipy.spatial.distance
            "chebyshev" = max, "cityblock" = L1, "minkowski" with p=
            or a function( Xvec, centervec ), e.g. Lqmetric below
        p: for minkowski metric -- local mod cdist for 0 < p < 1 too
        verbose: 0 silent, 2 prints running distances
    out:
        centers, k x dim
        clusters: each X -> its nearest center, ints N -> k
        distances, N
    see also: kmeanssample below, class Kmeans below.
    """
    if not issparse(X):
        X = np.asanyarray(X)  # ?
    centers = centers.todense() if issparse(centers) \
        else centers.copy()
    N, dim = X.shape
    k, cdim = centers.shape
    if dim != cdim:
        raise ValueError( "kmeans: X %s and centers %s must have the same number of columns" % (
            X.shape, centers.shape ))
    if verbose:
        print("kmeans: X %s  centers %s  delta=%.2g  maxiter=%d  metric=%s" % (
            X.shape, centers.shape, delta, maxiter, metric) )
    allx = np.arange(N)
    prevdist = 0
    for jiter in range( 1, maxiter+1 ):
        D = cdist_sparse( X, centers, metric=metric, p=p )  # |X| x |centers|
        xtoc = D.argmin(axis=1)  # X -> nearest center
        distances = D[allx,xtoc]
        avdist = distances.mean()  # median ?
        if verbose >= 2:
            print( "kmeans: av |X - nearest center| = %.4g" % avdist )
        if (1 - delta) * prevdist <= avdist <= prevdist \
        or jiter == maxiter:
            break
        prevdist = avdist
        for jc in range(k):  # (1 pass in C)
            c = np.where( xtoc == jc )[0]
            if len(c) > 0:
                centers[jc] = X[c].mean( axis=0 )
    if verbose:
        print("kmeans: %d iterations  cluster sizes:" % jiter, np.bincount(xtoc) )
    if verbose >= 2:
        r50 = np.zeros(k)
        r90 = np.zeros(k)
        for j in range(k):
            dist = distances[ xtoc == j ]
            if len(dist) > 0:
                r50[j], r90[j] = np.percentile( dist, (50, 90) )
        print( "kmeans: cluster 50 % radius", r50.astype(int) )
        print( "kmeans: cluster 90 % radius", r90.astype(int) )
            # scale L1 / dim, L2 / sqrt(dim) ?
    return centers, xtoc, distances

#...............................................................................
def kmeanssample( X, k, nsample=0, **kwargs ):
    """ 2-pass kmeans, fast for large N:
        1) kmeans a random sample of nsample ~ sqrt(N) from X
        2) full kmeans, starting from those centers
    """
        # merge w kmeans ? mttiw
        # v large N: sample N^1/2, N^1/2 of that
        # seed like sklearn ?
    N, dim = X.shape
    if nsample == 0:
        nsample = max( 2*np.sqrt(N), 10*k )
    Xsample = randomsample( X, int(nsample) )
    pass1centers = randomsample( X, int(k) )
    samplecenters = kmeans( Xsample, pass1centers, **kwargs )[0]
    return kmeans( X, samplecenters, **kwargs )

def cdist_sparse( X, Y, **kwargs ):
    """ -> |X| x |Y| cdist array, any cdist metric
        X or Y may be sparse -- best csr
    """
        # todense row at a time, v slow if both v sparse
    sxy = 2*issparse(X) + issparse(Y)
    if sxy == 0:
        return cdist( X, Y, **kwargs )
    d = np.empty( (X.shape[0], Y.shape[0]), np.float64 )
    if sxy == 2:
        for j, x in enumerate(X):
            d[j] = cdist( x.todense(), Y, **kwargs ) [0]
    elif sxy == 1:
        for k, y in enumerate(Y):
            d[:,k] = cdist( X, y.todense(), **kwargs ) [0]
    else:
        for j, x in enumerate(X):
            for k, y in enumerate(Y):
                d[j,k] = cdist( x.todense(), y.todense(), **kwargs ) [0]
    return d

def randomsample( X, n ):
    """ random.sample of the rows of X
        X may be sparse -- best csr
    """
    sampleix = random.sample( range( X.shape[0] ), int(n) )
    return X[sampleix]

def nearestcenters( X, centers, metric="euclidean", p=2 ):
    """ each X -> nearest center, any metric
            euclidean2 (~ withinss) is more sensitive to outliers,
            cityblock (manhattan, L1) less sensitive
    """
    D = cdist( X, centers, metric=metric, p=p )  # |X| x |centers|
    return D.argmin(axis=1)

def Lqmetric( x, y=None, q=.5 ):
    # yes a metric, may increase weight of near matches; see ...
    return (np.abs(x - y) ** q) .mean() if y is not None \
        else (np.abs(x) ** q) .mean()

#...............................................................................
class Kmeans:
    """ km = Kmeans( X, k= or centers=, ... )
        in: either initial centers= for kmeans
            or k= [nsample=] for kmeanssample
        out: km.centers, km.clusters, km.distances
        iterator:
            for jcenter, J in km:
                clustercenter = centers[jcenter]
                J indexes e.g. X[J], classes[J]
    """
    def __init__( self, X, k=0, centers=None, nsample=0, **kwargs ):
        self.X = X
        if centers is None:
            self.centers, self.clusters, self.distances = kmeanssample(
                X, k=k, nsample=nsample, **kwargs )
        else:
            self.centers, self.clusters, self.distances = kmeans(
                X, centers, **kwargs )

    def __iter__(self):
        for jc in range(len(self.centers)):
            yield jc, (self.clusters == jc)

#...............................................................................
if __name__ == "__main__":
    import random
    import sys
    from time import time

    N = 10000
    dim = 10
    ncluster = 10
    kmsample = 100  # 0: random centers, > 0: kmeanssample
    kmdelta = .001
    kmiter = 10
    metric = "euclidean"  # "chebyshev" = max, "cityblock" L1,  Lqmetric
    seed = 1

    exec( "\n".join( sys.argv[1:] ))  # run this.py N= ...
    np.set_printoptions( 1, threshold=200, edgeitems=5, suppress=True )
    np.random.seed(seed)
    random.seed(seed)

    print( "N %d  dim %d  ncluster %d  kmsample %d  metric %s" % (
        N, dim, ncluster, kmsample, metric) )
    X = np.random.exponential( size=(N,dim) )
        # cf scikits-learn datasets/
    t0 = time()
    if kmsample > 0:
        centers, xtoc, dist = kmeanssample( X, ncluster, nsample=kmsample,
            delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    else:
        randomcenters = randomsample( X, ncluster )
        centers, xtoc, dist = kmeans( X, randomcenters,
            delta=kmdelta, maxiter=kmiter, metric=metric, verbose=2 )
    print( "%.0f msec" % ((time() - t0) * 1000) )

    # also ~/py/np/kmeans/test-kmeans.py
