#
# LANS.py
#
# Implementation of the algorithm from the paper "Nonparametric sparsification
# of complex multiscale networks."
#
#
# Based on code by James M. Hughes (6/2010)
# Updated 11/2013 by Nick Foti

import numpy as N


def gen_similarity_matrix(feat, sigma=1):
    """ Generates a similarity matrix given the n x p numpy feature matrix
        (n rows in p dimensions) -- computes a distance matrix from the inputs,
        then a similarity matrix using shape parameter sigma for an exponential
        kernel of the distances (uses scipy.distance.pdist).

        feat : n x p numpy matrix
        sigma : variance parameter for exponential kernel
    """

    # input handling
    if not(type(feat) == N.matrix):
        raise TypeError('LANS: input matrix not of type numpy.matrix\n')

    # size of matrix
    (n, p) = feat.shape

    ## first, mean center the data
    dmean = N.mean(feat, axis=0)
    featmean = feat - N.tile(dmean, (n, 1))

    # now find data covariance matrix
    CM = featmean * featmean.T

    # divide by standard deviation
    covar = N.mat(N.diag(N.sqrt(CM))).T
    CM = CM / (covar*covar.T)

    # adjust for numerical issues
    #print CM[CM>1]
    #print CM[CM<-1]
    CM[CM > 1] = 1.
    CM[CM < -1] = -1.

    # now convert to distance
    DM = 2 * N.real(N.sin(N.arccos(CM)/2))
    # now make a similarity matrix
    S = N.exp(-DM/(2*sigma))

    return S, DM


def gen_ecdf_matrix(S):
    """ Creates a matrix in which each element is the ecdf value for that entry
        w.r.t. all other entries in its data row.

        N.B. does not subtract the ecdf value from 1; rather,
        comparison can be made by considering whether 1-alpha < lower tail
        probability (i.e., value of ecdf evaluated at x)

    """

    # do some input checking
    if type(S) != N.matrix:
        raise TypeError('gen_ecdf_matrix:  Invalid input type -- must be numpy.matrix')

    # now find the size of this matrix
    sz = S.shape

    # check for correct dimensions
    if sz[0] != sz[1]:
        raise ValueError('gen_ecdf_matrix:  Invalid input -- matrix is not square')

    # now make sure the matrix is of doubles
    S = N.double(S)

    # convenience renaming
    n = sz[0]

    ## at this point, we have a matrix of the correct size, which we can operate on
    # create the output adjancency matrix
    PVALmat = N.matrix(N.zeros((n,n)))

    # now loop through the rows of the network and construct the backbone network
    for i in range(0,n):
        # get the current row of the matrix, excluding the i'th value (since we don't want to
        # consider self-edges in such a network)
        idx = range(n)
        idx.remove(i)
        # actually grab the row
        currow = S[i,idx]

        # now, if the row is all zeros (shouldn't be the case, but might happen),
        # we don't want to consider it
        if N.sum(currow) > 0:
            currow = N.asarray(currow)

            # first we need to grab only the nonzero entries in this row
            idx = N.nonzero(currow)[1]

            # new length
            nn = len(idx)

            # get only the relevant entries
            currow = currow[:,idx]

            # compute probabilities of this row
            currow = currow / N.sum(currow)

            #currow = N.asarray(currow)

            # estimate the value of the empirical CDF of the edge weight probability
            # distribution at each of its values
            # N.B. 6/8/10 -- changed kind to 'weak' to reflect definition of CDF (i.e.,
            # prob. of RV taking on value less than or equal to input score)
            # TEST added 6/8/10 to improve speed yet again; uses repmat trick to do comparison
            # using matrices, based on following matlab code:
            #  sum(repmat(t', [1 length(t)]) >= repmat(t, [length(t) 1]),2) / length(t), where
            #   't' is the vector in question
            pvals = N.sum( N.tile(currow.T, (1, nn)) >= N.tile(currow, (nn, 1)), axis=1) / float(nn)
            if i == 0:
                print pvals
                print type(pvals)

            # PLACE probabilities back into matrix
            # NOTE:  here need to correct for indices that are greater than or equal to i
            #        since we removed the i'th entry in the row vector
            keep_idx = idx #N.asarray(range(len(pvals)))
    
            # now we need to adjust keep idx:  everywhere where then index is greater than
            # or equal to i, need to increment by 1
            adjidx = N.nonzero(keep_idx >= i)
            if len(adjidx) > 0:
                keep_idx[adjidx] = keep_idx[adjidx] + 1

            if i == 0:
                print adjidx
                print keep_idx
            
            # add pvalues to pval matrix (row by row)
            PVALmat[i,keep_idx] = pvals
            # "cancel out" the i'th value since we don't want self-edges
            PVALmat[i,i] = 0.

    # return the pval matrix
    return PVALmat


def backbone(S, alpha=0.05):
    """ Computes a backbone given a similarity matrix and significance value.
        Returns both a backbone and a CDF matrix.
    """

    # first, we'll generated a CDF matrix from
    CDFmat = gen_ecdf_matrix(S)

    # now create a backbone given alpha
    Abb = gen_backbone(CDFmat, alpha)

    Abb = N.multiply(Abb,S)

    return Abb, CDFmat


def backbone_from_cdf(CDFmat, alpha=0.05, S=None):
    """ Compute a backbone given a CDF matrix and significance value.
    """
    return gen_backbone(CDFmat, alpha, S)


def gen_backbone(CDFmat, alpha=0.05, S=None):
    """ Returns a backbone network given a CDF matrix and significance value
        and an optional similarity matrix for weights.
        Finds all entries in the matrix s.t. 1-alpha < CDF matrix entry
    """

    # do some input checking
    if type(CDFmat) != N.matrix:
        raise TypeError('gen_backbone:  Invalid input type -- must be numpy.matrix')

    # now find the size of this matrix
    sz = CDFmat.shape

    # check for correct dimensions
    if sz[0] != sz[1]:
        raise ValueError('gen_ecdf_matrix:  Invalid input -- matrix is not square')

    # now make sure the matrix is of doubles
    CDFmat = N.double(CDFmat)

    # convenience renaming
    n = sz[0]

    # now we need to find the entries 
    BBout = N.double(CDFmat > 1-alpha)

    # add weights if desired
    print type(S)
    if S != None:
        print '######## Adding weights to matrix...'
        BBout = N.multiply(BBout,S)
    else:
        print '######## NO weights specified...'

    return BBout


def cmdscale(D, ndim=2):
    """ Classical metric MDS.

        Port of Matlab implementation.
    """
    sz = N.shape(D)

    P = N.eye(sz[0]) - N.mat(N.tile(1/N.double(sz[0]), (sz[0], sz[0])))
    B = P * (-0.5 * N.multiply(D,D)) * P 
    E,V = N.linalg.eig((B+B.T) / 2.0)

    # get the eigenvalues
    e = E

    # sort descending
    sort_tuples = [(e[k],k) for k in range(len(e))]

    sort_tuples = sorted(sort_tuples, reverse=True)
    
    # grab sorted idx
    sidx = N.asarray([x[1] for x in sort_tuples])
 
    # grab eigenvalues
    e = N.asarray([x[0] for x in sort_tuples])

    # keep only positive eigenvalues 
    keep = N.where(e > N.max(N.abs(e)) * N.power(N.power(10.0,-15),0.75))[0]

    if len(keep) == 0:
        return N.zeros((sz[0],1))
    
    # get set of EVs
    V = V[:,sidx[keep]] * N.diag(N.sqrt(e[keep]),0)

    return D*V[:,:ndim]
