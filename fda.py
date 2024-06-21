import copy
import numpy as np
from locfdr import *


def smooth_basis(argvals,
                 y,
                 fdParobj,
                 wtvec=None,
                 fdnames=None,
                 covariates=None,
                 method='chol',
                 dfscale=1,
                 returnMatrix=False):
    # check y
    dimy = np.shape(y)
    ndy = len(dimy)
    n = dimy[0]
    # check argvals
    if argvals is None:
        raise ValueError("argvals required, is None.")
    dima = np.shape(argvals)
    if len(dima) == 1:
        dima = list(dima)
        dima.append(1)
        dima = tuple(dima)
    nda = len(dima)
    if ndy < nda:
        raise ValueError("argvals has ", nda, "dimensions ; y has only ", ndy)

    # check that the first dimensions of argvals and y match
    if dima[0] != dimy[0]:
        raise ValueError("Length of first dimensions of argvals and y do not match.")

    # select which version of smooth_basis to use, according to dim. of argvals
    # are all dimensions of argvals equal to the first nda of those of y?

    if nda < 3:
        # argvals is a matrix
        if dima[1] == 1:
            # argvals is a matrix with a single column, the usual case
            # the base version smooth_basis1 is called directly
            # see seperate file smooth_basis1 for this function

            sb2 = smooth_basis1(argvals, y, fdParobj, wtvec=wtvec, fdnames=fdnames, covariates=covariates,
                                method=method,
                                dfscale=dfscale, returnMatrix=returnMatrix)
            sb2.argvals = argvals
        else:
            sb2 = smooth_basis2(argvals, y, fdParobj, wtvec=wtvec, fdnames=fdnames, covariates=covariates,
                                method=method,
                                dfscale=dfscale, returnMatrix=returnMatrix)

        return sb2

    if nda < 4:
        return smooth_basis3(argvals, y, fdParobj, wtvec=wtvec, fdnames=fdnames, covariates=covariates,
                             method=method,
                             dfscale=dfscale, returnMatrix=returnMatrix)
    else:
        raise ValueError("Dimensions of argvals do not match those of y.")


def ycheck(y, n):
    y = np.matrix(y)
    ydim = y.shape
    if ydim[0] != n:
        raise ValueError("Y is not the same length as ARGVALS.")

    ndim = len(ydim)
    if ndim == 2:
        ncurve = ydim[1]
        nvar = 1

    if ndim == 3:
        ncurve = ydim[1]
        nvar = ydim[2]

    if ndim > 3:
        raise ValueError("Second argument must not have more than 3 dimensions.")

    return y, ncurve, nvar, ndim


def wtcheck(n, wtvec=None):
    # check n
    if np.round(n) != n:
        raise ValueError("n is not an integer.")
    if n < 1:
        raise ValueError("n is less than 1.")

    # check wtvec
    if wtvec is not None:
        dimw = np.shape(np.matrix(wtvec))
        if np.any(np.isnan(np.matrix(wtvec))):
            raise ValueError("WTVEC has NA values.")
        if np.all(dimw == n):
            # WTVEC is a matrix of order n
            onewt = False
            matwt = True
            # check weight matrix for being positive definite
            wteig = np.linalg.eig(wtvec)[0]
            if np.any(np.iscomplex(wteig)):
                raise ValueError("Weight matrix has complex eigenvalues.")
            if min(wteig) <= 0:
                raise ValueError("Weight matrix is not positive definite.")
        else:
            # WTVEC is treated as a vector
            if (len(dimw) > 1 and dimw[0] > 1 and dimw[1] > 1) or len(dimw) > 2:
                raise ValueError("WTVEC is neither a vector nor a matrix of order. n. ")
            wtvec = np.matrix(wtvec)
            if len(wtvec) == 1:
                wtvec = np.matrix(wtvec)
            if len(wtvec) != n:
                raise ValueError("WTVEC of wrong length.")
            if min(wtvec) <= 0:
                raise ValueError("Values in WTVEC are not positive.")
            onewt = False
            matwt = False
    else:
        wtvec = np.ones([n, 1])
        onewt = True
        matwt = False

    return wtvec, onewt, matwt


def smooth_basis1(argvals,
                  y,
                  fdParobj,
                  wtvec=None,
                  fdnames=None,
                  covariates=None,
                  method='chol',
                  dfscale=1,
                  returnMatrix=False):
    y = np.matrix(y)
    dimy = y.shape
    n = dimy[0]
    y, nrep, nvar, ndim = ycheck(y, n)
    ydim = y.shape

    # check ARGVALS

    # check fdParobj
    fdParobj = fdParcheck(fdParobj, nrep)
    fdobj = fdParobj.fd
    lamdba = fdParobj.lamdba
    Lfdobj = fdParobj.Lfd
    penmat = fdParobj.penmat

    # check lamdba
    if lamdba < 0:
        lamdba = 0

    # check WTVEC

    wtvec, onewt, matwt = wtcheck(n, wtvec)

    # extract information from fdParobj

    nderiv = Lfdobj.nderiv
    basisobj = fdobj.basisobj
    dropind = basisobj.dropind
    ndropind = len(dropind)
    nbasis = basisobj.nbasis - ndropind

    if ndim == 2:
        coef = np.zeros([nbasis, nrep])

    if ndim == 3:
        coef = np.zeros([nbasis, nrep, nvar])

    # check COVARIATES and set values for q, the number of covariates

    if covariates is not None:
        if np.shape(covariates)[0] != n:
            raise ValueError("smooth_basis_LS: covariates \n Optional argument COVARIATES has incorrect number of rows")
        q = np.shape(covariates)[1]
    else:
        q = 0
        beta = None

    # set up matrix of basis function values

    basismat = eval_basis(argvals, basisobj, 0, returnMatrix)

    if n > nbasis + q or lamdba > 0:
        # argument BASISMAT0 and BASISMAT by the covariate matrix
        # if it is supplied
        if covariates is not None:
            basismat = np.matrix(basismat)
            basismat = np.column_stack(basismat, np.zeros((basismat.shape[0], q)))
            basismat[0:n, nbasis: nbasis + q] = covariates

        # compute the product of the basis and weight matrix

        if matwt:
            wtfac = np.linalg.cholesky(wtvec)
            basisw = np.matmul(wtvec, basismat)
        else:
            rtwtvec = np.sqrt(wtvec)
            rtwtmat = np.ones([n, nrep])
            basisw = np.multiply(np.matmul(wtvec, np.ones([1, nbasis + q])), basismat)

        # the weight crossproduct of the basis matrix
        Bmat = np.matmul(transpose(basisw), basismat)
        Bmat0 = Bmat

        # set up right side of normal equations

        if ndim < 3:
            Dmat = np.matmul(transpose(basisw), y)
        else:
            Dmat = np.zeros([nbasis + q, nrep, nvar])
            for ivar in range(nvar):
                Dmat[:, :, ivar] = np.matmul(np.transpose(basisw), y[:, :, ivar])

        # if lamdba > 0:
        #     if penmat is None:
        #         penmat = eval_penalty(basisobj, Lfdobj)
        #     Bnorm = np.sqrt()
        penmat = None
        Bmat = Bmat0

        # Compute inverse of Bmat
        Bmat = (Bmat + transpose(Bmat)) / 2
        try:
            Lmat = np.linalg.cholesky(Bmat)
            Lmatinv = np.linalg.inv(Lmat)
            Bmatinv = np.matmul(Lmatinv, transpose(Lmatinv))
        except:
            Beig = np.linalg.eigvals(Bmat)
            BgoodEig = Beig[0] > 0
            Brank = sum(BgoodEig)
            goodVec = Beig[1][:, BgoodEig]
            Bmatinv = np.matmul(goodVec, np.multiply(Beig[0][BgoodEig], transpose(goodVec)))

        # compute coefficient matrix by solving normal equations
        Bmatinv = np.linalg.inv(Bmat)
        if ndim < 3:
            coef = np.matmul(Bmatinv, Dmat)
            if covariates is not None:
                beta = np.matrix(coef[nbasis: nbasis + q, :])
                coef = np.matrix(coef[0:nbasis, :])
            else:
                beta = None
        else:
            coef = np.zeros([nbasis, nrep, nvar])
            if covariates is not None:
                beta = np.zeros([q, nrep, nvar])
            else:
                beta = None
            for ivar in range(nvar):
                coefi = np.matmul(Bmatinv, Dmat[:, :, ivar])
                if covariates is not None:
                    beta[:, :, ivar] = coefi[nbasis:nbasis + q, :]
                    coef[:, :, ivar] = coefi[0:nbasis, :]
                else:
                    coef[:, :, ivar] = coefi
    else:
        if n == nbasis + q:
            # code for n == nbasis, q == 0 and lamdba == 0

            if ndim == 2:
                coef = np.linalg.solve(basismat, y)
            else:
                for ivar in range(nvar):
                    coef[0:n, :, ivar] = np.linalg.solve(basismat, y[:, :, ivar])
            penmat = None
        else:
            raise ValueError("The number of basis functions = ", nbasis + q, " exceeds ", n,
                             " = the number of points to be smoothed.")

    # compute SSE, yhatm GCV and other fit summaries

    # compute map from y to c
    if onewt:
        temp = np.matmul(transpose(basisw), basismat)
        if lamdba > 0:
            temp = temp + lamdba * penmat
        L = np.linalg.cholesky(temp)
        MapFac = np.linalg.solve(transpose(L), transpose(basismat))
        y2cMap = np.linalg.solve(L, MapFac)
    else:
        if matwt:
            temp = np.matmul(np.matmul(transpose(basismat), wtvec), basismat)
        else:
            temp = np.multiply(np.matmul(transpose(basismat), np.matrix(wtvec)), basismat)

        if lamdba > 0:
            temp = temp + lamdba * penmat

        L = np.linalg.cholesky((temp + transpose(temp)) / 2)
        MapFac = np.linalg.solve(transpose(L), transpose(basismat))
        if matwt:
            y2cMap = np.linalg.solve(L, np.matmul(MapFac, wtvec))
        else:
            y2cMap = np.linalg.solve(L, np.multiply(MapFac, np.repeat(np.matrix(wtvec), np.shape(MapFac)[0])))

    # compute degrees of freedom of smooth
    df = sum(np.diag(np.matmul(y2cMap, basismat)))

    # compute error sum of squares
    if ndim < 3:
        yhat = np.matmul(basismat[:, 0: nbasis], coef)
        SSE = np.power(y[0:n] - yhat, 2).sum()
    else:
        SSE = 0
        yhat = np.zeros([n, nrep, nvar])
        for ivar in range(nvar):
            yhat[:, :, ivar] = np.matmul(basismat[:, 0: nbasis], coef[:, :, ivar])
            SSE = SSE + np.power(y[0: n, :, ivar] - yhat[:, :, ivar], 2).sum()

        # compute GCV index
    if df is not None and df < n:
        if ndim < 3:
            gcv = np.zeros(nrep)
            for i in range(nrep):
                SSEi = np.power(y[0:n] - yhat[:, i], 2).sum()
                gcv[i] = pow((SSEi / n) / ((n - df) / n), 2)
        else:
            gcv = np.zeros([nrep, nvar])
            for ivar in range(nvar):
                for i in range(nrep):
                    SSEi = np.power(y[0:n, i, ivar] - yhat[:, i, ivar], 2).sum()
                    gcv[i, ivar] = pow((SSEi / n) / ((n - df) / n), 2)
    else:
        gcv = None

    if ndim < 3:
        coef = np.matrix(coef)
        fdobj = fd(coef[0: nbasis, :], basisobj, fdnames)
    else:
        fdobj = fd(coef[0: nbasis, :, :], basisobj, fdnames)

    if penmat is not None and covariates is not None:
        penmat = penmat[0: nbasis, 0: nbasis]

    return fdSmooth(fdobj, df, gcv, beta, SSE, penmat, y2cMap, argvals)


class fdSmooth:
    def __init__(self, fdobj, df, gcv, beta, SSE, penmat, y2cMap, argvals):
        self.fd = fdobj
        self.df = df
        self.gcv = gcv
        self.beta = beta
        self.SSE = SSE
        self.penmat = penmat
        self.y2cMap = y2cMap
        self.argvals = argvals


def smooth_basis2():
    pass


def smooth_basis3():
    pass


def smooth_basisPar(argval, y, fdobj=None, Lfdobj=None, lamdba=0, estimate=True, penmat=None, wtvec=None, fdnames=None,
                    covariates=None, method='chol', dfscale=1):
    # 1. check fdobj
    if fdobj is None:
        fdobj = create_bspline_basis(argval)
    else:
        if isinstance(fdobj, np.ndarray):
            if len(fdobj) == 1:
                if np.round(fdobj) != fdobj:
                    raise ValueError("'fdobj' is numeric but not an integer.")
                if np.sum(fdobj < 0) > 0:
                    raise ValueError("some of 'fdobj' is not positive.")
                fdobj = create_bspline_basis(argval, norder=fdobj)
            else:
                fdobj = fd(fdobj)

    # 2. fdPar: setup the functional parameter object from arguments

    fdP = fdPar(fdobj, Lfdobj=Lfdobj, lamdba=lamdba, estimate=estimate, penmat=penmat)

    # 3. smooth_basis: carry out smoothing by a call to smooth_basis and return the smoothlist object that this function returns

    return smooth_basis(argval, y, fdP, wtvec=wtvec, fdnames=fdnames,
                        covariates=covariates, method='chol', dfscale=dfscale)


class linmodList:
    def __init__(self, beta0estfd, beta1estbifd, yhatfdobj, coefvec = None, Dmat = None, Cmat = None):
        self.beta0estfd = beta0estfd
        self.beta1estbifd = beta1estbifd
        self.yhatfdobj = yhatfdobj
        self.coefvec = coefvec
        self.Dmat = Dmat
        self.Cmat = Cmat


def linmod(xfdobj, yfdobj, betalist, wtvec=None):
    if not isinstance(xfdobj, fd):
        raise ValueError("XFD is not a functional data object,")

    if not isinstance(yfdobj, fd):
        raise ValueError("YFD is not a functional data object,")

    ybasis = yfdobj.basisobj
    ynbasis = ybasis.nbasis
    ranget = ybasis.rangeval

    xbasis = xfdobj.basisobj
    ranges = xbasis.rangeval

    nfine = np.max((201, 10 * ynbasis + 1))
    tfine = np.linspace(ranget[0], ranget[1], nfine)

    # get dimensions of data

    coefy = yfdobj.coef
    coefx = xfdobj.coef
    coefdx = np.shape(coefx)
    coefdy = np.shape(coefy)
    ncurves = coefdx[1]
    if coefdy[1] != ncurves:
        raise ValueError("Number of obervations in first two arguments do not match.")

    # set up or check weight vector

    if wtvec is not None:
        wtvec = wtcheck(ncurves, wtvec)

    # get basis parameter objects

    if not isinstance(betalist, list):
        raise ValueError("betalist is not a list object.")

    if len(betalist) != 2:
        raise ValueError("betalist not of length 2.")

    alphafdPar = betalist[0]
    betabifdPar = betalist[1]

    if not isinstance(alphafdPar, fdPar):
        raise ValueError("BETACELL[0] is not a fdPar object.")

    if not isinstance(betabifdPar, bifdPar):
        raise ValueError("BETACELL[1] is not a bifdPar object.")

    # get Lfd objects

    alphaLfd = alphafdPar.Lfd
    betasLfd = betabifdPar.Lfds
    betatLfd = betabifdPar.Lfdt

    # get smoothing parameters

    alphalamdba = alphafdPar.lamdba
    betaslamdba = betabifdPar.lamdbas
    betatlamdba = betabifdPar.lamdbat

    # get basis object
    alphafd = alphafdPar.fd
    alphabasis = alphafd.basisobj
    alpharange = alphabasis.rangeval

    if (alpharange[0] != ranget[0]) or (alpharange[1] != ranget[1]):
        raise ValueError("Range of ALPHAFD coefficient and YFD not compatible.")

    betabifd = betabifdPar.bifd

    betasbasis = betabifd.sbasis
    betasrange = betasbasis.rangeval
    if (betasrange[0] != ranges[0]) or (betasrange[1] != ranges[1]):
        raise ValueError("Range of BETASFD coefficient and XFD not compatible.")

    betatbasis = betabifd.tbasis
    betatrange = betatbasis.rangeval
    if (betatrange[0] != ranget[0]) or (betatrange[1] != ranget[1]):
        raise ValueError("Range of BETATFD coefficient and YFD not compatible.")

    # get numbers of basis functions

    alphanbasis = alphabasis.nbasis
    betasnbasis = betasbasis.nbasis
    betatnbasis = betatbasis.nbasis

    # get inner product of basis functions and data functions

    Finprod = inprod(ybasis, alphabasis)
    Ginprod = inprod(ybasis, betatbasis)
    Hinprod = inprod(xbasis, betasbasis)

    ycoef = yfdobj.coef
    xcoef = xfdobj.coef
    Fmat = np.matmul(np.transpose(ycoef), Finprod)
    Gmat = np.matmul(np.transpose(ycoef), Ginprod)
    Hmat = np.matmul(np.transpose(xcoef), Hinprod)

    if wtvec is None:
        HHCP = np.matmul(np.transpose(Hmat), Hmat)
        HGCP = np.matmul(np.transpose(Hmat), Gmat)
        H1CP = np.matrix(np.sum(Hmat, axis=0))
        F1CP = np.matrix(np.sum(Fmat, axis=0))
    else:
        HHCP = np.matmul(np.transpose(Hmat), np.multiply(np.outer(wtvec, betasnbasis), Hmat))
        HGCP = np.matmul(np.transpose(Hmat), np.multiply(np.outer(wtvec, betatnbasis), Gmat))
        H1CP = np.matmul(np.transpose(Hmat), wtvec)
        F1CP = np.matmul(np.transpose(Fmat), wtvec)

    # get inner product of basis functions

    alphattmat = inprod(alphabasis, alphabasis)
    betalttmat = inprod(betatbasis, alphabasis)
    betassmat = inprod(betasbasis, betasbasis)
    betattmat = inprod(betatbasis, betatbasis)

    # get penalty matrix
    if alphalamdba > 0:
        alphapenmat = eval_penalty(alphabasis, alphaLfd)
    else:
        alphapenmat = 0

    if betaslamdba > 0:
        betaspenmat = eval_penalty(betasbasis, betasLfd)
    else:
        betaspenmat = 0

    if betatlamdba > 0:
        betatpenmat = eval_penalty(betatbasis, betatLfd)
    else:
        betatpenmat = 0

    # set up coefficient matrix and right side fro stationary equations

    betan = betasnbasis * betatnbasis
    ncoef = alphanbasis + betan
    Cmat = np.zeros([ncoef, ncoef])
    Dmat = np.zeros([ncoef, 1])

    # row for alpha

    ind11 = 0
    ind12 = alphanbasis
    ind21 = ind11
    ind22 = ind12
    Cmat[ind11:ind12, ind21:ind22] = ncurves * alphattmat
    if alphalamdba > 0:
        Cmat[ind11:ind12, ind21:ind22] = Cmat[ind11:ind12, ind21:ind22] + alphalamdba * alphapenmat
    ind21 = alphanbasis
    ind22 = alphanbasis + betan
    Cmat[ind11:ind12, ind21:ind22] = np.kron(H1CP, betalttmat)

    Dmat[ind11:ind12] = np.transpose(F1CP)

    # row for data
    ind11 = alphanbasis
    ind12 = alphanbasis + betan
    ind21 = 0
    ind22 = alphanbasis
    Cmat[ind11:ind12, ind21:ind22] = np.transpose(Cmat[ind21:ind22, ind11:ind12])
    ind21 = ind11
    ind22 = ind12
    Cmat[ind11:ind12, ind21:ind22] = np.kron(HHCP, betattmat)
    if betaslamdba > 0:
        Cmat[ind11:ind12, ind21:ind22] = Cmat[ind11:ind12, ind21:ind22] + betaslamdba * np.kron(betaspenmat, betattmat)

    if betatlamdba > 0:
        Cmat[ind11:ind12, ind21:ind22] = Cmat[ind11:ind12, ind21:ind22] + betatlamdba * np.kron(betassmat, betatpenmat)

    Dmat[ind11:ind12] = HGCP.reshape((betan, 1))

    # solve the equations
    coefvec = np.linalg.solve(Cmat, Dmat)

    # set up the coefficient function estimates
    # functional structure for the alpha function

    ind11 = 0
    ind12 = alphanbasis
    alphacoef = coefvec[ind11:ind12]
    alphafd = fd(alphacoef, alphabasis)

    # bi-functional structure for the beta function
    ind11 = alphanbasis
    ind12 = betan + alphanbasis
    betacoef = np.transpose(np.matrix(coefvec[ind11:ind12]).reshape((betatnbasis, betasnbasis)))
    betafd = bifd(np.transpose(betacoef), betasbasis, betatbasis)

    # functional data structure for the yhat functions
    xbetacoef = np.dot(betacoef, np.transpose(Hmat))
    xbetafd = fd(xbetacoef, betatbasis)
    yhatmat = np.matmul(eval_fd(tfine, alphafd), np.ones([1, ncurves])) + eval_fd(tfine, xbetafd)
    yhatfd = smooth_basis(tfine, yhatmat, ybasis).fd

    linmodres = linmodList(beta0estfd=alphafd, beta1estbifd=betafd, yhatfdobj=yhatfd, coefvec = coefvec, Cmat = Cmat, Dmat = Dmat)
    return linmodres


def predit_linmod(linmodres, newdata=None):
    if newdata is None:
        return linmodres.yhatfdobj
    else:
        xbasis = newdata.basisobj
        xnbasis = xbasis.nbasis
        ranget = xbasis.rangeval
        coefx = newdata.coef
        coefdx = np.shape(coefx)
        ncurves = coefdx[1]

        nfine = np.max((201, 10 * xnbasis + 1))
        tfine = np.linspace(ranget[0], ranget[1], nfine)

        alphafd = linmodres.beta0estfd
        betasbasis = linmodres.beta1estbifd.sbasis
        Hinprod = inprod(xbasis, betasbasis)
        xcoef = coefx
        Hmat = np.dot(np.transpose(xcoef), Hinprod)
        betacoef = np.transpose(linmodres.beta1estbifd.coef)
        xbetacoef = np.dot(betacoef, np.transpose(Hmat))
        xbetafd = fd(xbetacoef, linmodres.beta1estbifd.tbasis)
        temp = eval_fd(tfine, alphafd)
        if temp.shape[1] != 1:
            temp = np.transpose(temp)
        yhatmat = np.dot(temp, np.ones([1, ncurves])) + eval_fd(tfine, xbetafd)
        return smooth_basis(tfine, yhatmat, xbasis).fd


def functionalBoosting(dataXFunctional, scalarFd, betalistScalar, dataYFunctional, argvals, fdParobjX, fdParobjY, betalist, boost_control, step_len, duplicates_sample, duplicates_learner):

    # xShape[0] and yShape[0] are not necessary the same, xShape[1] == yShape[1]
    xShape = list(dataXFunctional.shape) # two dimensions, first is num of observations, second is num of samples
    yShape = list(dataYFunctional.shape) # two dimensions, first is num of observations, second is num of samples

    if xShape[1] != yShape[1]:
        raise ValueError("Nums of samples are not the same between x and y.")

    # Here, to test the performance of Ray, duplicate the sample size by @para: duplicates_sample
    newDataXFunctional = np.asarray([x for i in range(duplicates_sample) for x in dataXFunctional.T]).T
    newDataYFunctional = np.asarray([x for i in range(duplicates_sample) for x in dataYFunctional.T]).T

    scalarFd[0] = np.asarray([x for i in range(duplicates_sample) for x in scalarFd[0]]).reshape([xShape[1] * duplicates_sample, 1])

    for i in range(1, len(scalarFd)):
        scalarFd[i].coef = np.asarray([x for j in range(duplicates_sample) for x in scalarFd[i].coef]).reshape([1, xShape[1] * duplicates_sample])

    xFdobjFunctional = smooth_basis(argvals, newDataXFunctional, fdParobjX).fd
    yFdobjFunctional = smooth_basis(argvals, newDataYFunctional, fdParobjY).fd

    result = []

    # Step 1: use the mean value of yFdobjFunctional
    init = yFdobjFunctional.mean()
    coefdim = np.shape(yFdobjFunctional.coef)
    init.coef = np.repeat(init.coef, repeats=coefdim[1], axis=1)
    residual = yFdobjFunctional - init
    result.append([-1, init])

    # Starting from step 2
    i = 2
    while i <= boost_control:

        # Here, to test the performance of Ray, duplicate the function-on-function baselinear size by @para: duplicates_learner
        model_scalar = fRegress(residual, scalarFd, betalistScalar)
        sse = inprod(((model_scalar.yhatfdobj - residual) * (model_scalar.yhatfdobj - residual))).sum()
        resid = [sse]
        model = [model_scalar]
        for j in range(duplicates_learner):
            model_function = linmod(xFdobjFunctional, residual, betalist)
            sse = inprod(((model_function.yhatfdobj - residual) * (model_function.yhatfdobj - residual))).sum()
            model.append(model_function)
            resid.append(sse)

        best = np.argmin(resid)
        bestmodel = model[best]

        result.append([best, bestmodel])
        # print("Iteration ", i, "Best model is ", best)
        residual = yFdobjFunctional - pred_gradboost1(result, step_len)
        i = i + 1

    return result


def pred_gradboost1(res, step_length):
    length = len(res)
    y_pred = res[0]
    for i in range(1, length):
        y_pred = step_length * res[i].yhatfdobj + y_pred
    return y_pred


def pred_gradboost2(res, x_function1, x_function2, step_len):
    length = len(res)
    coefdim = np.shape(x_function1.coef)
    res[0][1].coef = np.repeat(res[0][1].coef, repeats=coefdim[1], axis=1)

    y_pred = res[0][1]
    for i in range(1, length):
        if res[i][0] == 0:
            y_pred = step_len * predit_linmod(res[i][1], x_function1) + y_pred
        else:
            y_pred = step_len * predit_linmod(res[i][1], x_function2) + y_pred

    return y_pred


class fd:
    def __init__(self, coef=None, basisobj=None, fdnames=None):

        # # check coef and get its dimension
        # if not isinstance(coef, np.ndarray) and not isinstance(coef , int):
        #     raise ValueError("'coef' is not an instance of numpy ndarray, please check.")
        #
        # if coef is None:
        #     coef = np.zeros([basisobj.nbasis])
        #
        # coef = np.matrix(coef)
        # coefd = coef.shape
        # ndim = len(coefd)
        #
        # if ndim > 3:
        #     raise ValueError("'coef' not of dimension 1, 2 or 3")
        #
        # # check basisobj
        #
        # nbasis = basisobj.nbasis
        # dropind = basisobj.dropind
        # ndropind = len(dropind)
        #
        # if coefd[0] != nbasis - ndropind:
        #     raise ValueError("First dim. of 'coef' not equal to 'nbasis - ndropind'.")
        #
        # # setup number of replicates and number of variables
        # if ndim > 1:
        #     nrep = coefd[1]
        # else:
        #     nrep = 1
        #
        # if ndim > 2:
        #     nvar = coefd[2]
        # else:
        #     nvar = 1

        self.coef = coef
        self.basisobj = basisobj
        self.fdnames = fdnames

    def __add__(self, other):
        if isinstance(other, fd):
            return fd(self.coef + other.coef, self.basisobj, self.fdnames)
        else:
            pass

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, fd):
            coef = self.coef - other.coef
            return fd(coef, self.basisobj, self.fdnames)

    def __mul__(self, other):
        if isinstance(other, fd):
            coef1 = self.coef
            coef2 = other.coef
            coefd1 = np.shape(coef1)
            coefd2 = np.shape(coef2)
            ndim1 = len(coefd1)
            ndim2 = len(coefd2)

            if len(coefd1) != len(coefd2):
                raise ValueError("Number of dimensions of coefficient arrays does not match.")
            # allow for one function having a single replicate
            # and if so, copy it as many times as there are replicates
            # in the other function

            # self is single, other has replications

            if coefd1[1] == 1 and coefd2[1] > 1:
                if ndim1 == 2:
                    coef1 = np.matrix(coef1).reshape(coefd1[0], coefd2[1])
                elif ndim1 == 3:
                    temp = np.zeros(coefd2)
                    for j in range(coefd1[2]):
                        temp[:, :, j] = np.outer(coef1[:, 0, j], np.ones(coefd2[1]))
                    coef1 = temp
                else:
                    raise ValueError("Dimensions of coefficient matrices not compatable.")
                coefd1 = np.shape(coef1)
                self.coef = coef1

            # self has replications, other is single

            if coefd1[1] > 1 and coefd2[1] == 1:
                if ndim2 == 2:
                    coef2 = np.matrix(coef2).reshape(coefd2[0], coefd1[1])
                elif ndim1 == 3:
                    temp = np.zeros(coefd1)
                    for j in range(coefd2[2]):
                        temp[:, :, j] = np.outer(coef2[:, 0, j], np.ones(coefd1[1]))
                    coef2 = temp
                else:
                    raise ValueError("Dimensions of coefficient matrices not compatible.")
                coefd2 = np.shape(coef2)
                other.coef = coef2

            # check that numbers of replications are equal
            if coefd1[1] != coefd2[1]:
                raise ValueError("Number of replications are not equal.")

            # check for matching in the muktivariate case

            if ndim1 > 2 and ndim2 > 2 and ndim1 != ndim2:
                raise ValueError("Both arguments multivariate, but involve different numbers of functions.")

            # extract the two bases

            basisobj = self.basisobj * other.basisobj
            nbasis1 = self.basisobj.nbasis
            nbasis2 = other.basisobj.nbasis
            rangeval = self.basisobj.rangeval

            # set up a fine mesh for evaluating the product
            neval = max(10 * max(nbasis1, nbasis2) + 1, 201)
            evalarg = np.linspace(rangeval[0], rangeval[1], neval)

            # set up arrays for function values
            fdarray1 = eval_fd(evalarg, self)
            fdarray2 = eval_fd(evalarg, other)

            # compute product arrays

            if (ndim1 <= 2 and ndim2 <= 2) or (ndim1 > 2 and ndim2 > 2):
                fdarray = np.multiply(fdarray1, fdarray2)
            else:
                # product array where the number of dimensions don't match
                if ndim1 == 2 and ndim2 > 2:
                    fdarray = np.zeros(coefd2)
                    for ivar in range(coefd2[2]):
                        fdarray[:, :, ivar] = fdarray1 * fdarray2[:, :, ivar]
                if ndim1 > 2 and ndim2 == 2:
                    fdarray = np.zeros(coefd1)
                    for ivar in range(coefd1[2]):
                        fdarray[:, :, ivar] = fdarray1[:, :, ivar] * fdarray2

            # set up the coefficient by projecting on to the
            # product basis

            coefprod = project_basis(fdarray, evalarg, basisobj)
        else:
            fac = other
            fac = np.matrix(fac)
            coef = self.coef
            coefd = np.shape(coef)
            coefprod = np.multiply(fac, coef)
            basisobj = self.basisobj

        fdprod = fd(coefprod, basisobj)

        return fdprod

    __rmul__ = __mul__

    def mean(self):
        coef = self.coef
        coefd = np.shape(coef)
        ndim = len(coefd)
        basisobj = self.basisobj
        nbasis = basisobj.nbasis
        dropind = basisobj.dropind
        ndropind = len(dropind)
        if ndim == 2:
            coefmean = np.reshape(coef.mean(1), (nbasis - ndropind, 1))
        else:
            nvar = coefd[2]
            coefmean = np.zeros((coefd[0], 1, nvar))
            for j in range(nvar):
                coefmean[:, 0, j] = coef[:, :, j].mean(1)
        meanfd = fd(coefmean, basisobj)
        return meanfd

    def sum(self):
        coef = self.coef
        coefd = np.shape(coef)
        ndim = len(coefd)
        basisobj = self.basisobj
        nbasis = basisobj.nbasis
        dropind = basisobj.dropind
        ndropind = len(dropind)

        if ndim == 2:
            if nbasis - ndropind == 0:
                coefsum = np.matrix(np.sum(coef, axis=1))
            else:
                coefsum = np.matrix(np.sum(coef, axis=1)).reshape((nbasis - ndropind, 1))
        else:
            nvar = coefd[2]
            coefsum = np.zeros([coefd[0], 1, nvar])
            for j in range(nvar):
                coefsum[:, 1, j] = np.sum(coef[:, :, j], axis=1)

        sumfd = fd(coefsum, basisobj)
        return sumfd


def eval_basis(evalarg, basisobj, Lftobj=0, returnMatrix=False):
    # 1. check
    if (isinstance(basisobj, int) or isinstance(basisobj, list)) and isinstance(evalarg, basis):
        temp = basisobj
        basisobj = evalarg
        evalarg = temp

    # check basisobj
    if not isinstance(basisobj, basis):
        raise ValueError("Second argument is not a basis object.")

    # check LFDOBJ

    Lftobj = int2Lfd(Lftobj)

    # 2. Set up
    # determine the highest order of derivative NDERIV required

    nderiv = Lftobj.nderiv

    # get weight coefficient functions

    bwtlist = Lftobj.bwlist

    # 3. Do
    # get highest order of basis matrix

    basismat = getbasismatrix(evalarg, basisobj, nderiv, returnMatrix)

    # Compute the weightes combination of derivations is
    # evaluated here if the operator is not defined by an integer
    # and the order of derivative is positive

    if nderiv > [0]:
        nbasis = np.shape(basismat)[1]
        oneb = np.ones([1, nbasis])
        nonintwrd = False
        for j in range(nderiv[0]):
            bfd = bwtlist[j]
            bbasis = bfd.basisobj
            if bbasis.type != 'const' or bfd.coef != 0:
                nonintwrd = True
        if nonintwrd:
            for j in range(nderiv[0]):
                bfd = bwtlist[j]
                temp = bfd.coef == 0.0
                if not temp.all():
                    wjarry = eval_fd(evalarg, bfd, 0, returnMatrix)
                    Dbasismat = getbasismatrix(evalarg, basisobj, [j - 1], returnMatrix)
                    basismat = basismat + np.multiply(np.dot(wjarry, oneb), Dbasismat)

    # if (not returnMatrix) and len(np.shape(basismat)) == 2:
    #     return basismat.tolist()
    # else:
    #     return basismat
    return basismat


def eval_fd(evalarg, fdobj, Lfdobj=0, returnMatrix=False):
    Lfdobj = int2Lfd(Lfdobj)
    if ((isinstance(fdobj, int) or isinstance(fdobj, np.ndarray)) or isinstance(fdobj, float)) and isinstance(evalarg,
                                                                                                              fd):
        temp = fdobj
        fdobj = evalarg
        evalarg = temp

    # check EVALARG
    Evalarg = evalarg
    # if not isinstance(evalarg, int):
    #     evalarg = int(Evalarg)
    #     nNa = sum(np.isnan(evalarg))
    #     if nNa > 0:
    #         raise ValueError("as.numeric(evalarg) contains", nNa, "NA")

    evaldim = np.shape(evalarg)
    if len(evaldim) >= 3:
        raise ValueError("Argument 'evalarg' is not a vector or a matrix.")

    # check FDOBJ
    if not isinstance(fdobj, fd):
        raise ValueError("Argument FD is not a functional data object.")

    basisobj = fdobj.basisobj
    nbasis = basisobj.nbasis
    rangeval = basisobj.rangeval
    onerow = [1] * nbasis

    # temp = [i for i in evalarg if not np.isnan(i)]
    # EPS = 5 * sys.float_info.epsilon
    # if min(temp) < rangeval[0] - EPS or max(temp) > rangeval[1] + EPS:
    #     warnings.warn("Values in argument 'evalarg' are outside of permitted range and will be ignored.")

    # get maximum number of evaluation values
    if isinstance(evalarg, np.ndarray):
        n = len(evalarg)
    elif isinstance(evalarg, float):
        evalarg = [evalarg]
        n = len(evalarg)
    else:
        n = evaldim[0]

    # set up coefficient array for FD
    coef = fdobj.coef
    coef = np.array(coef)
    coefd = np.shape(coef)
    ndim = len(coefd)
    nrep = 1 if ndim <= 1 else coefd[1]
    nvar = 1 if ndim <= 2 else coefd[2]

    # check coef is conformable with evalarg
    if len(evaldim) > 1:
        if evaldim[1] != 1 and evaldim[1] != coefd[1]:
            raise ValueError("'evalarg' has ", evaldim[1], "columns; does not match ", ndim[1],
                             " = number of columns of fdobj.coefs.")

    # Set up array for function values
    if ndim <= 2:
        evalarray = np.zeros([n, nrep])
    else:
        evalarray = np.zeros((n, nrep, nvar))

    # case where EVALARG is a vector of values to be used for all curves
    evalarg = np.array(evalarg)
    if len(evaldim) <= 1:
        # evalarg[evalarg < rangeval[0]  - 1e-10] = np.nan
        # evalarg[evalarg > rangeval[1] + 1e-10] = np.nan
        basismat = eval_basis(evalarg, basisobj, Lfdobj, returnMatrix)
        # print(evalarg)
        # evaluate the functions at arguments in EVALARG

        if ndim <= 2:
            evalarray = np.dot(basismat, coef)
        else:
            evalarray = np.zeros([n, nrep, nvar])
            for i in range(nvar):
                evalarray[:, :, i] = np.dot(basismat, coef[:, :, i])
    else:
        # case of evaluation values varying from curve to curve
        for i in range(nrep):
            evalargi = evalarg[:, i]
            basismat = eval_basis(evalargi, basisobj, Lfdobj, returnMatrix)
            if ndim == 2:
                evalarray[:, i] = np.dot(basismat, coef[:, i])
            elif ndim == 3:
                for j in range(nvar):
                    evalarray[:, i, j] = np.dot(basismat, coef[:, i, j])

    return np.matrix(evalarray)


class fdname:
    def __init__(self, time, reps, values):
        self.args = time
        self.reps = reps
        self.funs = values


def getbasismatrix(evalarg, basisobj, nderiv=[0], returnMatrix=False):
    if (isinstance(basisobj, int) or isinstance(basisobj, list)) and isinstance(evalarg, basis):
        temp = basisobj
        basisobj = evalarg
        evalarg = temp

    # check EVALARG
    if evalarg is None:
        raise ValueError("evalarg required; is None.")
    Evalarg = evalarg

    # check BASISOBJ
    if not isinstance(basisobj, basis):
        raise ValueError("Second argument is not a basis object.")

    # Search for stored basis matrix and return it if found
    if not (basisobj.basisvalues is None or len(basisobj.basisvalues) == 0):
        # one or more stored basis matrix found,
        # check that requested derivative is available
        basisvalues = basisobj.basisvalues
        if not isinstance(basisvalues, list):
            raise ValueError("BASISVALUES is not a list.")
        nvalues = len(basisvalues)
        N = len(evalarg)
        OK = False
        for ivalues in range(nvalues):
            basisvaluesi = basisvalues[ivalues]
            if len(basisvaluesi) >= nderiv[0] + 2:
                if N == len(basisvaluesi):
                    if all(basisvaluesi == evalarg):
                        basismat = basisvaluesi[nderiv[0] + 2]
                        OK = True

        if OK:
            return basismat
        # if len(np.shape(basismat)) == 2:
        #     return basismat

    # Compute the basis matrix and return it
    # Extract information about the basis
    type = basisobj.type
    nbasis = basisobj.nbasis
    params = basisobj.params
    rangeval = basisobj.rangeval
    dropind = basisobj.dropind

    # select basis and evaluate it at EVALARG values

    #   B-Spline Basis
    if type == 'bspline':
        if len(params) == 0:
            breaks = [rangeval[0], rangeval[1]]
        else:
            breaks = [rangeval[0]] + params.tolist() + [rangeval[1]]
        norder = nbasis - len(breaks) + 2
        basismat = bsplineS(evalarg, breaks, norder, nderiv)
    elif type == 'const':
        basismat = np.ones([len(evalarg), 1])
    else:
        raise ValueError("Only constant and bspline basis are supported now. Basis type not recognized.")
    basismat = np.array(basismat)

    return basismat


def bsplineS(x, breaks, norder=4, nderiv=0, returnMatrix=False):
    x = np.array(x)
    n = len(x)
    tol = 1e-14
    nbreaks = len(breaks)
    if nbreaks < 2:
        raise ValueError("Number of knots less than 2.")
    if min(np.diff(np.array(breaks))) < 0:
        raise ValueError("Knots are not increasing.")

    if max(x) > (max(breaks) + tol) or min(x) < (min(breaks) - tol):
        raise ValueError("Knots do not span the values of 'X'.")

    if x[n - 1] > breaks[nbreaks - 1]:
        breaks[nbreaks - 1] = x[n - 1]
    if x[0] < breaks[0]:
        breaks[0] = x[0]

    if norder > 20:
        raise ValueError("NORDER exceeds 20.")
    if norder < 1:
        raise ValueError("NORDER less than 1.")
    if nderiv[0] > 19:
        raise ValueError("NDERIV exceeds 19.")
    if norder < 0:
        raise ValueError("NDERIV less than 0.")
    if nderiv[0] >= norder:
        raise ValueError("NDERIV cannot be as large as order of B-Spline.")

    knots = [breaks[0]] * (norder - 1) + breaks + [breaks[nbreaks - 1]] * (norder - 1)
    deriv = [nderiv[0]] * n
    nbasis = nbreaks + norder - 2

    if nbasis >= norder:
        basismat = splineDesign(knots, x, norder, deriv[0])
        return basismat
    else:
        raise ValueError("NBASIS is less than NORDER.")


def create_constant_basis(rangeval=[0, 1], names='const', axes=None):
    if len(rangeval) == 1:
        if rangeval[0] <= 0:
            raise ValueError("RANGEVAL a single value that is not positive. ")
        rangeval = [0, rangeval[0]]

    type = names
    nbasis = 1
    params = [0]
    dropind = [0]
    quadvals = [0]
    values = []
    basisvalues = []

    basisobj = basis(type=type, rangeval=rangeval, nbasis=nbasis, params=params, dropind=dropind, quadvals=quadvals,
                     values=values, basisvalues=basisvalues)
    return basisobj


def int2Lfd(m=[0]):
    if isinstance(m, Lfd):
        Lfdobj = m
        return Lfdobj

    if isinstance(m, int):
        m = [m]
        # raise ValueError("Argument not numeric and not a linear differential operator.")

    if len(m) != 1:
        raise ValueError("Argument is not a scalar.")

    if round(m[0]) != m[0]:
        raise ValueError("Argument is not an integer.")

    if m[0] < 0:
        raise ValueError("Argument is negative.")

    if m[0] == 0:
        bwtlist = None
    else:
        basisobj = create_constant_basis([0, 1])
        bwtlist = []
        for j in range(m[0]):
            bwtlist.append(fd(np.zeros(0), basisobj))

    Lfdobj = Lfd(m, bwtlist)
    return Lfdobj


def fd2list(fdobj):
    # get the coefficient matrix and the basis
    coef = fdobj.coef
    coefsize = np.shape(coef)
    nrep = coefsize[1]

    # check whether FDOBJ is univariate

    if len(coefsize) > 2:
        raise ValueError("FDOBJ is not univariate.")

    fdlist = []
    for i in range(nrep):
        fdlist.append(fd(coef[:, i], fdobj.basisobj, fdobj.fdnames))

    return fdlist


class fdPar:
    def __init__(self, fdobj=None, Lfdobj=None, lamdba=0, estimate=True, penmat=None):
        # default fdPar object
        if not isinstance(fdobj, fd):
            if fdobj is None:
                fdobj = fd()
            else:
                if isinstance(fdobj, basis):
                    nbasis = fdobj.nbasis
                    dropind = fdobj.dropind
                    nbasis = nbasis - dropind if len(dropind) != 0 else nbasis
                    coefs = np.zeros([nbasis, nbasis])
                    # fdnames = ['time', 'reps 1', 'values']
                    # if fdobj.name is not None:
                    #     basisnames = [i for i in fdobj.names if fdobj.names.index(i) not in dropind] if len(
                    #         dropind) > 0 else fdobj.names
                    #     fdnames[1] = basisnames
                    fdobj = fd(coefs, fdobj, fdnames=None)
                elif isinstance(fdobj, int) or isinstance(fdobj, np.ndarray) or isinstance(fdobj, list):
                    fdobj = fd(basisobj=fdobj)
                else:
                    raise ValueError("First argument is neither a functional data object nor a basis object.")
        else:
            nbasis = fdobj.basisobj.nbasis

        # check parameters
        # check Lfobj
        if Lfdobj is None:
            norder = norder_bspline(fdobj.basisobj) if fdobj.basisobj.type == 'bspline' else 2
            Lfdobj = int2Lfd(max(0, norder - 2))
        else:
            Lfdobj = int2Lfd(Lfdobj)

        if not isinstance(Lfdobj, Lfd):
            raise ValueError("'Lfobj' is not a linear differential operator object.")

        # check lambda

        if not isinstance(lamdba, int):
            raise ValueError("Class of LAMBDA is not numeric.")
        if lamdba < 0:
            raise ValueError("LAMBDA is negative.")

        # check estimate

        if not isinstance(estimate, bool):
            raise ValueError("Class of ESIMATE is not logical.")

        # check penmat

        if penmat is not None:
            penmatsize = np.shape(penmat)
            if any(i != nbasis for i in penmatsize):
                raise ValueError("Dimensions of PENMAT are not correct.")

        # setup attributes

        self.fd = fdobj
        self.Lfd = Lfdobj
        self.lamdba = lamdba
        self.estimate = estimate
        self.penmat = penmat


def fdParcheck(fdParobj, ncurve=None):
    if isinstance(fdParobj, basis) and ncurve is None:
        raise ValueError("First argument is basis object and second argument is missing.")
    if not isinstance(fdParobj, fdPar):
        if isinstance(fdParobj, fd):
            fdParobj = fdPar(fdParobj)
        if isinstance(fdParobj, basis):
            nbasis = fdParobj.nbasis
            fdParobj = fdPar(fd(np.zeros([nbasis, ncurve]), fdParobj))
        else:
            raise ValueError(
                "'fdParobj' is not a functional parameter object, not a functional data object and not a basis object.")

    return fdParobj


class Lfd:
    def __init__(self, nderiv=[0], bwtlist=[]):
        if round(nderiv[0]) != nderiv[0]:
            raise ValueError("Order of operator is not an integer.")
        if nderiv[0] < 0:
            raise ValueError("Order of operator is negative.")

        # check that bwtlist is either a list or a fd object

        if not isinstance(bwtlist, list) and not isinstance(bwtlist, fd) and bwtlist is not None:
            raise ValueError("BWTLIST is neither a LIST or a FD object.")

        # if bwtlist is missing or NULL, convert it to a constant basis FD object

        if bwtlist is None:
            bwtlist = []
            if nderiv[0] > 0:
                conbasis = create_constant_basis()
                for j in range(nderiv[0]):
                    bwtlist.append(fd(coef=0, basisobj=conbasis))

        # if BWTLIST is a fd object, convert to a list object

        if isinstance(bwtlist, fd):
            bwtlist = fd2list(bwtlist)

        # check size of bwtlist

        nbwt = len(bwtlist)

        if nbwt != nderiv[0] and nbwt != nderiv[0] + 1:
            raise ValueError("The size of bwtlist inconsistent with NDERIV.")

        # check individual list entries for class
        # and find a default range

        if nderiv[0] > 0:
            rangevec = [0, 1]
            for j in range(nbwt):
                bfdj = bwtlist[j]

                if isinstance(bfdj, fdPar):
                    bfdj = bfdj.fd
                    bwtlist[j] = bfdj

                if not isinstance(bfdj, fd) and not isinstance(bfdj, int):
                    raise ValueError("AN element of BWTLIST contains something other than an fd object or an integer.")

                if isinstance(bfdj, fd):
                    bbasis = bfdj.basisobj
                    rangevec = bbasis.rangeval
                else:
                    if len(bfdj) == 1:
                        bwtfd = fd(bfdj, conbasis)
                        bwtlist[j] = bwtfd
                    else:
                        raise ValueError("An element of BWTLIST conatins a more than one integer.")

            # check that the ranges are compatible

            for j in range(nbwt):
                bfdj = bwtlist[j]
                if isinstance(bfdj, fdPar):
                    bfdj = bfdj.fd
                bbasis = bfdj.basisobj
                btype = bbasis.type
                if btype != 'const':
                    brange = bbasis.rangeval
                    if rangevec != brange:
                        raise ValueError("Ranges are not compatible.")

        self.nderiv = nderiv
        self.bwlist = bwtlist


def norder_bspline(basisobj):
    return basisobj.nbasis - len(basisobj.params)


def project_basis(y, argvals, basisobj, penalize=False):
    if not isinstance(basisobj, basis):
        raise InputError("Third argument BASISOBJ is not a basis object.")
    basismat = getbasismatrix(argvals, basisobj, [0])
    Bmat = np.matmul(np.transpose(basismat), basismat)

    if penalize:
        penmat = eval_penalty(basisobj)
        penmat = penmat + 1e-10 * max(penmat) * np.identity(np.shape(penmat)[0])
        lamdba = (0.0001 * sum(np.diagonal(Bmat))) / sum(np.diagonal(Bmat))
        Cmat = Bmat + lamdba * penmat
    else:
        Cmat = Bmat

    if len(np.shape(y)) <= 2:
        Dmat = transpose(basismat) * y
        coef = np.linalg.solve(Cmat, Dmat)
    else:
        nvar = np.shape(y)[2]
        coef = np.zeros([basisobj.nbasism, np.shape(y)[1], nvar])
        for ivar in range(nvar):
            Dmat = transpose(basismat) * y[:, :, ivar]
            coef[:, :, ivar] = np.linalg.solve(Cmat, Dmat)

    return coef


def eval_penalty(basisobj, Lfdobj=int2Lfd(0), rng=[0, 1]):
    if isinstance(basisobj, fd):
        basisobj = basisobj.basisobj

    if isinstance(basisobj, fdPar):
        fdobj = basisobj.fd
        basisobj = fdobj.basisobj

    if not isinstance(basisobj, basis):
        raise ValueError("Argument BASISOBJ is not a functional basis object.")

    # set up default values
    rangeval = basisobj.rangeval

    # deal with the case where LFDOBJ is an integer
    LFdobj = int2Lfd(Lfdobj)

    # determine basis type
    type = basisobj.type

    # choose appropriate penalty matrix function
    if type == 'bspline':
        penaltymat = bsplinepen(basisobj, Lfdobj, rangeval)
    elif type == 'const':
        rangeval = getbasisrange(basisobj)
        penaltymat = rangeval[1] - rangeval[0]
    else:
        raise ValueError("Basis type not reconizable, cannot find penalty matrix.")

    dropind = basisobj.dropind
    nbasis = basisobj.nbasis

    if len(dropind) > 0:
        index = range(nbasis)
        index = [i for i in index if index.index(i) not in dropind]
        penaltymat = penaltymat[index, index]

    return penaltymat


def getbasisrange(basisobj):
    if not isinstance(basisobj, basis):
        raise ValueError("'basisobj' is not a functional basis object.")

    rangeval = basisobj.rangeval
    return rangeval


def bsplinepen(basisobj, Lfdobj=2, returnMatrix=False):
    pass
    # if not isinstance(basisobj, basis):
    #     raise ValueError("First argument is not a basis object.")
    # rng = basisobj.rangeval
    # # check basis type
    # type = basisobj.type
    #
    # # check LFDOBJ
    # Lfdobj = int2Lfd(Lfdobj)
    #
    # # get basis information
    # nbasis = basisobj.nbasis
    # params = basisobj.params
    #
    # # if there are no internal knots, use the monomial penalty
    # if len(params) == 0:
    #     basisobj = create_monomial_basis(basisobj.rangeval, nbasis, np.array(range(nbasis - 1)))
    #     penaltymatrix = monomialpen(basisobj, Lfdobj, rng)
    #     return penaltymatrix
    #
    # # normal case
    # # check
    # nNA = sum(np.isnan(rng))
    # if nNA > 0:
    #     raise ValueError("rng constains NA.")
    # nNAp = sum(np.isnan(params))
    # if nNAp > 0:
    #     raise ValueError("params constains NA")
    #
    # breaks = [rng[0]] + params + [rng[1]]
    # nbreaks = len(breaks)
    # ninterval = nbreaks - 1
    #
    # # check break values
    # if len(breaks) > 2:
    #     raise ValueError("The length of argument breaks is less than 2.")
    #
    # # find the highest order derivative on LFD
    # nderiv = Lfdobj.nderiv
    # norder = nbasis - len(params)
    #
    # # check for order of derivative being equal or greater than order of spline
    #
    # if nderiv >= norder:
    #     raise ValueError("Derivative of order cannot be taken for B-spline of order. \n",
    #                      "Probable cause is a value of the argument.\n",
    #                      "in function create.basis.fd that is too small.")
    #
    # # check for order of derivative being equal to order of spline
    # # minus one, in which case following code won't work
    #
    # if nderiv > 0 and nderiv == norder - 1:
    #     raise ValueError("Penalty matrix cannot be evaluated for derivative of order", nderiv,
    #                      " for B-spline of order ", norder)
    #
    # # special case where LFD is D^NDERIV and NDERIV = NORDER -1
    # bwtlist = Lfdobj.bwlist
    # isintLfd = True
    #
    # if nderiv > 0:
    #     for ideriv in range(nderiv):
    #         fdj = bwtlist[ideriv]
    #         if fdj is not None:
    #             if any(fdj.coefs != 0):
    #                 isintLfd = False
    #                 breaks
    #
    # if isintLfd and nderiv == norder - 1:
    #     halfseq = (breaks[1:(nbreaks - 1)] + breaks[0:(nbreaks - 2)]) / 2
    #     halfmat = bsplineS(halfseq, breaks, norder, nderiv, returnMatrix)
    #     brwidth = np.diff(breaks)
    #     penaltymat = np.matmul(np.matmul(transpose(halfmat), np.diag(breaks)),halfmat)
    #     return penaltymat
    #
    # # look for knot multiplicities within the range
    #
    # intbreaks = [rng[0]]  + params +[rng[1]]
    # index =  (intbreaks )


class basis:

    def __init__(self, type='bspline', rangeval=[0, 1], nbasis=2, params=[], dropind=[], quadvals=[], values=[],
                 basisvalues=[], names=None):

        # check type
        if type == 'bspline' or type == 'Bspline' or type == 'spline' or type == 'Bsp' or type == 'bsp':
            self.type = 'bspline'
        elif type == 'con' or type == 'const' or type == 'constant':
            self.type = 'const'
        elif type == 'exp' or type == 'expon' or type == 'exponential':
            self.type = 'expon'
        else:
            raise ValueError(
                "@param type error, only b-spline, exponential and constant basis functions are supported now.")

        # check rangeval
        # rangeval should be a numpy array of length 2 containing the lower and upper boundaries

        if len(rangeval) != 2:
            raise ValueError("rangeval should be a numpy array of length 2 containing the lower and upper boundaries")
        elif rangeval[1] <= rangeval[0]:
            raise ValueError("Argument rangeval is not strictly increasing.")
        else:
            self.rangeval = rangeval

        # check nbasis

        if nbasis <= 0:
            raise ValueError("Argument nbasis is not positive.")
        elif round(nbasis) != nbasis:
            raise ValueError("Argument basis is not an integer.")
        else:
            self.nbasis = nbasis

        # checl quadvals

        if len(quadvals) != 0 and quadvals is not None:
            dim = np.shape(quadvals)
            self.nquad = dim[0]
            self.ncol = 1
            # if self.nquad == 2 and self.ncol > 2:
            #     quadvals = transpose(quadvals)
            #     self.nquad = quadvals.shape[0]
            #     ncol = quadvals.shape[1]
            # if self.nquad < 2:
            #     raise ValueError("Less than two quadrature points are supplied.")
            # if ncol != 2:
            #     raise ValueError("'quadvals' does not have two columns.")

        # check VALUES is present, and set to a single empty list if not.

        if len(values) != 0 and values is not None:
            if values[0] != self.nquad:
                raise ValueError("Number of rows in 'values' not equal to number of quadrature points")
            if values[1] != self.nbasis:
                raise ValueError("Number of columns in 'values' not equal to number of basis functions")
        else:
            values = []

        # check BASISVALUES is present, and set to list() if not
        # If present, it must be a two-dimensional list created by a command like
        # listobj = np.array([2,3])

        if len(basisvalues) != 0 and basisvalues is not None:
            sizeves = np.shape(basisvalues)
            if len(sizeves) != 2:
                raise ValueError("BASISVALUES is not 2-dimensional.")
            # Waiting to check
            # for (i in 1:sizevec[1]) {
            # if (length(basisvalues[[i, 1]]) != dim(basisvalues[[i, 2]])[1]) stop(
            # paste("Number of argument values not equal number",
            # "of values."))
            # }
        else:
            basisvalues = None

        self.basisvalues = basisvalues

        # check if DROPIND is presentm and set to default if not

        if len(dropind) > 0:
            ndrop = len(dropind)
            if ndrop > self.nbasis:
                raise ValueError("Too many index values in DROPIND.")
            dropind.sort()

        self.dropind = dropind
        # Waiting to check
        # if (ndrop > 1 & & any(diff(dropind)) == 0)
        #     stop('Multiple index values in DROPIND.')
        # for (i in 1:ndrop) {
        # if (dropind[i] < 1 | | dropind[i] > nbasis)
        # stop('A DROPIND index value is out of range.')
        # }

        # check values
        # nvalues = length(values)
        # if (nvalues > 0 & & length(values[[1]] > 0)) {
        # for (ivalue in 1:nvalues) {
        #     derivvals = values[[ivalue]]
        # derivvals = derivvals[, -dropind]
        # values[[ivalue]] = derivvals
        # }
        # }
        # }
        self.values = values

        # select the appropriate type and process

        if self.type == 'const':
            self.params = 0
        elif self.type == 'bspline':
            if params is not None:
                nparams = len(params)
                if nparams > 0:
                    if params[0] < self.rangeval[0]:
                        raise ValueError("Smallest value in BREAKS not within RANGEVAL")
                    if params[nparams - 1] >= self.rangeval[1]:
                        raise ValueError("Largest value in BREAKS not within RANGEVAL")

        self.params = params

        self.names = names

    def summary(self):

        print("\n Type: ", self.type, "\n")
        print("\n Range: ", self.rangeval[1], "to ", self.rangeval[2], "\n")
        if self.type != 'const':
            print("\n Number of basis functions: ", self.nbasis, "\n")
        if len(self.dropind) > 0:
            print(len(self.dropind), "indices of basis functions to be dropped.")

    def __eq__(self, other):

        type1 = self.type
        range1 = self.rangeval
        nbasis1 = self.nbasis
        pars1 = self.params
        drop1 = self.dropind

        type2 = other.type
        range2 = other.rangeval
        nbasis2 = other.nbasis
        pars2 = other.params
        drop2 = other.dropind

        if type1 != type2:
            return False

        if range1[0] != range2[0] or range1[1] != range2[1]:
            return False

        if nbasis1 != nbasis2:
            return False

        if np.all(drop1 != drop2) != 0:
            return False

        return True

    def __mul__(self, other):
        # Important!!!
        # Currently, this method only support constant and bspline basis
        range1 = self.rangeval
        range2 = other.rangeval

        if range1[0] != range2[0] or range1[1] != range2[1]:
            raise ValueError("Ranges are not equal.")

        # deal with constant bases

        type1 = self.type
        type2 = other.type

        if type1 == 'const' and type2 == 'const':
            prodbasisobj = create_constant_basis(range1)
            return prodbasisobj

        if type1 == 'const':
            return other

        if type2 == 'const':
            return self

        # deal with bspline basis
        # get the number of basis functions
        nbasis1 = self.nbasis
        nbasis2 = other.nbasis
        if type1 == 'bspline' and type2 == 'bspline':

            interiorknots1 = self.params
            interiorknots2 = other.params

            interiorknots12 = np.union1d(interiorknots1, interiorknots2)
            interiorknots12.sort()
            nunique = len(interiorknots12)
            multunique = np.zeros(nunique)

            for i in range(nunique):
                mult1 = interiorknots1 == (interiorknots12[i]) if len(interiorknots1) > 0 else 0
                mult2 = interiorknots2 == (interiorknots12[i]) if len(interiorknots2) > 0 else 0
                multunique[i] = max(sum(mult1), sum(mult2))

            allknots = np.zeros(int(np.sum(multunique)))

            m2 = 0
            for i in range(nunique):
                m1 = m2 + 1
                m2 = int(m2 + multunique[i])
                allknots[m1 - 1:m2] = interiorknots12[i]

            norder1 = nbasis1 - len(interiorknots1)
            norder2 = nbasis2 - len(interiorknots2)
            # norder is not allowed to exceed 20
            norder = min([norder1 + norder2 - 1, 20])
            allbreaks = np.concatenate((np.concatenate((range1[0], allknots), axis=None), range1[1]), axis=None)
            nbasis = len(allbreaks) + norder - 2
            prodbasisobj = create_bspline_basis(rangeval=range1, nbasis=nbasis, norder=norder, breaks=allbreaks)

            return prodbasisobj

        if type1 == 'bspline' or type2 == 'bspline':
            norder = 8
            if type1 == 'bspline':
                interiorknots1 = self.params
                norder1 = nbasis1 - len(interiorknots1)
                norder = min([norder1 + 2, norder])
            if type2 == 'bspline':
                interiorknots2 = other.params
                norder2 = nbasis2 - len(interiorknots2)
                norder = min([norder2 + 2, norder])

            nbasis = max([nbasis1 + nbasis2, norder + 1])
            prodbasisobj = create_bspline_basis(rangeval=range1, nbasis=nbasis, norder=norder)

            return prodbasisobj

    __rmul__ = __mul__


def create_bspline_basis(rangeval=[], nbasis=4, norder=4, breaks=[], dropind=[], quadvals=[], values=[],
                         basisvalues=[], names='bspline'):
    type = names

    # check RANGEVAL
    # 1.1 first check breaks is either None or is numeric with positive length

    if breaks is not None and len(breaks) != 0:
        if min(np.diff(breaks)) < 0:
            raise ValueError("One or more breaks differences are negative.")
        if len(breaks) < 1:
            breaks = []
        if sum(np.isnan(breaks)) > 0:
            raise ValueError("breaks contains NAs; not allowed")
        if sum(np.isinf(breaks)) > 0:
            raise ValueError("breaks contains Infs; not allowed")

    if len(rangeval) < 1:
        if len(breaks) == 0:
            rangeval = [0, 1]
        else:
            rangeval = [min(breaks), max(breaks)]
            if np.diff(rangeval) == 0:
                raise ValueError("diff(range(breaks)) == 0, not allowed.")
    else:
        nNa = sum(np.isnan(rangeval))
        if nNa > 0:
            raise ValueError("@param rangeval contains NA, not allowed.")

    if len(rangeval) == 1:
        if rangeval <= 0:
            raise ValueError("'rangeval' a single value is not positive, is", rangeval)
        rangeval = np.array([0, rangeval])

    if len(rangeval) > 2:
        if len(breaks) != 0:
            raise ValueError("breaks can not be provided with len(rangeval) > 2")
        breaks = rangeval
        rangeval = np.array([min(breaks), max(breaks)])

    if rangeval[0] >= rangeval[1]:
        raise ValueError("rangeval[0] must be less than rangeval[1].")

    # 2. check norder

    if norder <= 0 or norder % 1 > 0:
        raise ValueError("norder must be a single positive integer.")

    # 3. check nbasis

    nbreaks = len(breaks)
    if nbasis is not None:
        if nbasis <= 0 or nbasis % 1 > 0:
            raise ValueError("nbasis must be a single positive integer.")
        elif nbasis < norder:
            raise ValueError("nbasis must be at least norder")
        # 4. check breaks
        if len(breaks) != 0:
            if nbreaks < 2:
                raise ValueError("Number of values in argument 'breaks' less than 2.")
            if breaks[0] != rangeval[0] or breaks[nbreaks - 1] != rangeval[1]:
                raise ValueError("Range if argument 'breaks' not identical to that of argument 'rangeval'.")
            if min(np.diff(breaks)) < 0:
                raise ValueError("values in argument 'breaks' are decreasing.")
            if nbasis != norder + nbreaks - 2:
                raise ValueError("Relation nbasis = norder + length(breaks) -2 does not hold.")
        else:
            breaks = np.linspace(rangeval[0], rangeval[1], nbasis - norder + 2)
            nbreaks = len(breaks)
    else:
        if len(breaks) == 0:
            nbasis = norder
        else:
            nbasis = len(breaks) + norder - 2

    # 5. Set up the PARAMS vector, which contains only the interior knots
    if nbreaks > 2:
        params = breaks[1: (nbreaks - 1)]
    else:
        params = []

    # 6. set up basis object

    basisobj = basis(type=type, rangeval=rangeval, nbasis=nbasis, params=params, dropind=dropind, quadvals=quadvals,
                     values=values,
                     basisvalues=basisvalues)

    # 7. names

    return basisobj


class bifd:
    def __init__(self, coef=np.zeros([2, 1]), sbasisobj=create_bspline_basis(), tbasisobj=create_bspline_basis(),
                 fdname=None):

        if not isinstance(coef, np.ndarray):
            raise ValueError("coef must be numerical vector or matrix.")
        elif len(np.shape(coef)) == 1:
            raise ValueError("Argument coef is not at least 2 dimensional.")
        else:
            coefd = np.shape(coef)
            ndim = len(coefd)

        if ndim > 4:
            raise ValueError("Argument COEF is not correct.")

        # check SBASISOBJ

        if not isinstance(sbasisobj, basis):
            raise ValueError("Argument SBASISOBJ is not of basis class.")

        if coefd[0] != sbasisobj.nbasis:
            raise ValueError("Number of coefficients does not match number of basis functions from SBASISOBJ.")

        # check TBASISOBJ

        if not isinstance(tbasisobj, basis):
            raise ValueError("Argument "
                             "TBASISOBJ is not of basis class.")

        if coefd[1] != tbasisobj.nbasis:
            raise ValueError("Number of coefficients does not match number of basis functions from TBASISOBJ.")

        # set up number of replicates and number of variables

        if ndim > 2:
            nrep = coefd[2]
        else:
            nrep = 1

        if ndim > 3:
            nvar = coefd[3]
        else:
            nvar = 1

        self.coef = coef
        self.sbasis = sbasisobj
        self.tbasis = tbasisobj
        self.bifdname = fdname


class bifdPar:
    def __init__(self, bifdobj, Lfdobjs=int2Lfd(2), Lfdobjt=int2Lfd(2), lamdbas=0, lamdbat=0, estimate=True):

        if not isinstance(bifdobj, bifd):
            raise ValueError("BIFDOBJ is not a bivariate functional data object.")

        # check the linear differential operators

        LFdobjs = int2Lfd(Lfdobjs)
        LFdobjt = int2Lfd(Lfdobjt)

        if not isinstance(LFdobjs, Lfd):
            raise ValueError("LFDOBJS is not a linear differential operator object.")

        if not isinstance(LFdobjt, Lfd):
            raise ValueError("LFDOBJT is not a linear differential operator object.")

        # check the roughness penalty parameters

        if not isinstance(lamdbas, int):
            raise ValueError("LAMDBAS is not numeric.")

        if lamdbas < 0:
            lamdbas = 0

        if not isinstance(lamdbat, int):
            raise ValueError("LAMDBAT is not numeric.")

        if lamdbat < 0:
            lamdbat = 0

        if not isinstance(estimate, bool):
            raise ValueError('ESTIMATE is not logical.')

        # set up the bifdPar object

        self.bifd = bifdobj
        self.estimate = estimate
        self.lamdbas = lamdbas
        self.lamdbat = lamdbat
        self.Lfds = Lfdobjs
        self.Lfdt = Lfdobjt


#########
# Important!!!
# Currently, only support constant and bspline basis
#########
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
#  This function creates a bspline functional data basis.
#  Arguments
#  RANGEVAL...an array of length 2 containing the lower and upper
#             boundaries for the rangeval of argument values,
#             or a positive number, in which case command
#             rangeval <- c(0, rangeval) is executed.
#             the default is c(0,1)
#  NBASIS  ...the number of basis functions.  This argument must be
#             supplied, and must be a positive integer.
#  NORDER  ...order of b-splines (one higher than their degree).  The
#             default of 4 gives cubic splines.
#  BREAKS  ...also called knots, these are a non-decreasing sequence
#             of junction points between piecewise polynomial segments.
#             They must satisfy BREAKS[1] = RANGEVAL[1] and
#             BREAKS[NBREAKS] = RANGEVAL[2], where NBREAKS is the total
#             number of BREAKS.  There must be at least 2 BREAKS.
#  There is a potential for inconsistency among arguments NBASIS, NORDER,
#  and BREAKS since
#             NBASIS = NORDER + LENGTH(BREAKS) - 2
#  An error message is issued if this is the case.  Although previous
#  versions of this function attempted to resolve this inconsistency in
#  various ways, this is now considered to be too risky.
#  DROPIND ...A vector of integers specifiying the basis functions to
#             be dropped, if any.  For example, if it is required that
#             a function be zero at the left boundary, this is achieved
#             by dropping the first basis function, the only one that
#             is nonzero at that point.
#  QUADVALS...A NQUAD by 2 matrix.  The firs t column contains quadrature
#             points to be used in a fixed point quadrature.  The second
#             contains quadrature weights.  For example, for (Simpson"s
#             rule for (NQUAD = 7, the points are equally spaced and the
#             weights are delta.*[1, 4, 2, 4, 2, 4, 1]/3.  DELTA is the
#             spacing between quadrature points.  The default is
#             matrix("numeric",0,0).
#  VALUES ... A list, with entries containing the values of
#             the basis function derivatives starting with 0 and
#             going up to the highest derivative needed.  The values
#             correspond to quadrature points in QUADVALS and it is
#             up to the user to decide whether or not to multiply
#             the derivative values by the square roots of the
#             quadrature weights so as to make numerical integration
#             a simple matrix multiplication.
#             Values are checked against QUADVALS to ensure the correct
#             number of rows, and against NBASIS to ensure the correct
#             number of columns.
#             The default value of is VALUES is vector("list",0).
#             VALUES contains values of basis functions and derivatives at
#             quadrature points weighted by square root of quadrature weights.
#             These values are only generated as required, and only if slot
#             QUADVALS is not matrix("numeric",0,0).
#  BASISVALUES...A vector of lists, allocated by code such as
#             vector("list",1).
#             This field is designed to avoid evaluation of a
#             basis system repeatedly at a set of argument values.
#             Each list within the vector corresponds to a specific set
#             of argument values, and must have at least two components,
#             which may be tagged as you wish.
#             The first component in an element of the list vector contains the
#             argument values.
#             The second component in an element of the list vector
#             contains a matrix of values of the basis functions evaluated
#             at the arguments in the first component.
#             The third and subsequent components, if present, contain
#             matrices of values their derivatives up to a maximum
#             derivative order.
#             Whenever function getbasismatrix is called, it checks
#             the first list in each row to see, first, if the number of
#             argument values corresponds to the size of the first dimension,
#             and if this test succeeds, checks that all of the argument
#             values match.  This takes time, of course, but is much
#             faster than re-evaluation of the basis system.  Even this
#             time can be avoided by direct retrieval of the desired
#             array.
#             For example, you might set up a vector of argument values
#             called "evalargs" along with a matrix of basis function
#             values for these argument values called "basismat".
#             You might want too use tags like "args" and "values",
#             respectively for these.  You would then assign them
#             to BASISVALUES with code such as
#               basisobj$basisvalues <- vector("list",1)
#               basisobj$basisvalues[[1]] <-
#                               list(args=evalargs, values=basismat)
#  BASISFNNAMES ... Either a character vector of length NABASIS
#             or a single character string to which NORDER, "." and
#             1:NBASIS are appended by the command
#                paste(names, norder, ".", 1:nbreaks, sep="").
#             For example, if norder = 4, this defaults to
#                     'bspl4.1', 'bspl4.2', ... .
#  Returns
#  BASISFD ...a functional data basis object

#  Last modified  11 February 2015 by Jim Ramsay

#  -------------------------------------------------------------------------
#  Default basis for missing arguments:  A B-spline basis over [0,1] of
#    of specified norder with norder basis functions.
#    norder = 1 = one basis function = constant 1
#    norder = 2 = two basis functions = 2 right triangles,
#      one left, the other right.  They are a basis for straight lines
#      over the unit interval, and are equivalent to a monomial basis
#      with two basis functions.  This B-spline system can be
#      explicitly created with the command
#                create.bspline.basis(c(0,1), 2, 2)
#    norder = 3 = three basis functions:  x^2, x-(x-.5)^2, (x-1)^2
#    norder = 4 = default = 4 basis functions
#      = the simplest cubic spline basis
#  -------------------------------------------------------------------------

def create_bspline_basis(rangeval=[], nbasis=4, norder=4, breaks=[], dropind=[], quadvals=[], values=[],
                         basisvalues=[], names='bspline'):
    type = names

    # check RANGEVAL
    # 1.1 first check breaks is either None or is numeric with positive length

    if breaks is not None and len(breaks) != 0:
        if min(np.diff(breaks)) < 0:
            raise ValueError("One or more breaks differences are negative.")
        if len(breaks) < 1:
            breaks = []
        if sum(np.isnan(breaks)) > 0:
            raise ValueError("breaks contains NAs; not allowed")
        if sum(np.isinf(breaks)) > 0:
            raise ValueError("breaks contains Infs; not allowed")

    if len(rangeval) < 1:
        if len(breaks) == 0:
            rangeval = [0, 1]
        else:
            rangeval = [min(breaks), max(breaks)]
            if np.diff(rangeval) == 0:
                raise ValueError("diff(range(breaks)) == 0, not allowed.")
    else:
        nNa = sum(np.isnan(rangeval))
        if nNa > 0:
            raise ValueError("@param rangeval contains NA, not allowed.")

    if len(rangeval) == 1:
        if rangeval <= 0:
            raise ValueError("'rangeval' a single value is not positive, is", rangeval)
        rangeval = np.array([0, rangeval])

    if len(rangeval) > 2:
        if len(breaks) != 0:
            raise ValueError("breaks can not be provided with len(rangeval) > 2")
        breaks = rangeval
        rangeval = np.array([min(breaks), max(breaks)])

    if rangeval[0] >= rangeval[1]:
        raise ValueError("rangeval[0] must be less than rangeval[1].")

    # 2. check norder

    if norder <= 0 or norder % 1 > 0:
        raise ValueError("norder must be a single positive integer.")

    # 3. check nbasis

    nbreaks = len(breaks)
    if nbasis is not None:
        if nbasis <= 0 or nbasis % 1 > 0:
            raise ValueError("nbasis must be a single positive integer.")
        elif nbasis < norder:
            raise ValueError("nbasis must be at least norder")
        # 4. check breaks
        if len(breaks) != 0:
            if nbreaks < 2:
                raise ValueError("Number of values in argument 'breaks' less than 2.")
            if breaks[0] != rangeval[0] or breaks[nbreaks - 1] != rangeval[1]:
                raise ValueError("Range if argument 'breaks' not identical to that of argument 'rangeval'.")
            if min(np.diff(breaks)) < 0:
                raise ValueError("values in argument 'breaks' are decreasing.")
            if nbasis != norder + nbreaks - 2:
                raise ValueError("Relation nbasis = norder + length(breaks) -2 does not hold.")
        else:
            breaks = np.linspace(rangeval[0], rangeval[1], nbasis - norder + 2)
            nbreaks = len(breaks)
    else:
        if len(breaks) == 0:
            nbasis = norder
        else:
            nbasis = len(breaks) + norder - 2

    # 5. Set up the PARAMS vector, which contains only the interior knots
    if nbreaks > 2:
        params = breaks[1: (nbreaks - 1)]
    else:
        params = []

    # 6. set up basis object

    basisobj = basis(type=type, rangeval=rangeval, nbasis=nbasis, params=params, dropind=dropind, quadvals=quadvals,
                     values=values,
                     basisvalues=basisvalues)

    # 7. names

    return basisobj


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
def transpose(matrix):
    transposed = []
    for i in range(len(matrix[0])):
        transposed.append([row[i] for row in matrix])
    return transposed


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
class basis:

    def __init__(self, type='bspline', rangeval=[0, 1], nbasis=2, params=[], dropind=[], quadvals=[], values=[],
                 basisvalues=[], names=None):

        # check type
        if type == 'bspline' or type == 'Bspline' or type == 'spline' or type == 'Bsp' or type == 'bsp':
            self.type = 'bspline'
        elif type == 'con' or type == 'const' or type == 'constant':
            self.type = 'const'
        elif type == 'exp' or type == 'expon' or type == 'exponential':
            self.type = 'expon'
        else:
            raise ValueError(
                "@param type error, only b-spline, exponential and constant basis functions are supported now.")

        # check rangeval
        # rangeval should be a numpy array of length 2 containing the lower and upper boundaries

        if len(rangeval) != 2:
            raise ValueError("rangeval should be a numpy array of length 2 containing the lower and upper boundaries")
        elif rangeval[1] <= rangeval[0]:
            raise ValueError("Argument rangeval is not strictly increasing.")
        else:
            self.rangeval = rangeval

        # check nbasis

        if nbasis <= 0:
            raise ValueError("Argument nbasis is not positive.")
        elif round(nbasis) != nbasis:
            raise ValueError("Argument basis is not an integer.")
        else:
            self.nbasis = nbasis

        # checl quadvals

        if len(quadvals) != 0 and quadvals is not None:
            dim = np.shape(quadvals)
            self.nquad = dim[0]
            self.ncol = 1
            # if self.nquad == 2 and self.ncol > 2:
            #     quadvals = transpose(quadvals)
            #     self.nquad = quadvals.shape[0]
            #     ncol = quadvals.shape[1]
            # if self.nquad < 2:
            #     raise ValueError("Less than two quadrature points are supplied.")
            # if ncol != 2:
            #     raise ValueError("'quadvals' does not have two columns.")

        # check VALUES is present, and set to a single empty list if not.

        if len(values) != 0 and values is not None:
            if values[0] != self.nquad:
                raise ValueError("Number of rows in 'values' not equal to number of quadrature points")
            if values[1] != self.nbasis:
                raise ValueError("Number of columns in 'values' not equal to number of basis functions")
        else:
            values = []

        # check BASISVALUES is present, and set to list() if not
        # If present, it must be a two-dimensional list created by a command like
        # listobj = np.array([2,3])

        if len(basisvalues) != 0 and basisvalues is not None:
            sizeves = np.shape(basisvalues)
            if len(sizeves) != 2:
                raise ValueError("BASISVALUES is not 2-dimensional.")
            # Waiting to check
            # for (i in 1:sizevec[1]) {
            # if (length(basisvalues[[i, 1]]) != dim(basisvalues[[i, 2]])[1]) stop(
            # paste("Number of argument values not equal number",
            # "of values."))
            # }
        else:
            basisvalues = None

        self.basisvalues = basisvalues

        # check if DROPIND is presentm and set to default if not

        if len(dropind) > 0:
            ndrop = len(dropind)
            if ndrop > self.nbasis:
                raise ValueError("Too many index values in DROPIND.")
            dropind.sort()

        self.dropind = dropind
        # Waiting to check
        # if (ndrop > 1 & & any(diff(dropind)) == 0)
        #     stop('Multiple index values in DROPIND.')
        # for (i in 1:ndrop) {
        # if (dropind[i] < 1 | | dropind[i] > nbasis)
        # stop('A DROPIND index value is out of range.')
        # }

        # check values
        # nvalues = length(values)
        # if (nvalues > 0 & & length(values[[1]] > 0)) {
        # for (ivalue in 1:nvalues) {
        #     derivvals = values[[ivalue]]
        # derivvals = derivvals[, -dropind]
        # values[[ivalue]] = derivvals
        # }
        # }
        # }
        self.values = values

        # select the appropriate type and process

        if self.type == 'const':
            self.params = 0
        elif self.type == 'bspline':
            if params is not None:
                nparams = len(params)
                if nparams > 0:
                    if params[0] < self.rangeval[0]:
                        raise ValueError("Smallest value in BREAKS not within RANGEVAL")
                    if params[nparams - 1] >= self.rangeval[1]:
                        raise ValueError("Largest value in BREAKS not within RANGEVAL")

        self.params = params

        self.names = names

    def summary(self):

        print("\n Type: ", self.type, "\n")
        print("\n Range: ", self.rangeval[1], "to ", self.rangeval[2], "\n")
        if self.type != 'const':
            print("\n Number of basis functions: ", self.nbasis, "\n")
        if len(self.dropind) > 0:
            print(len(self.dropind), "indices of basis functions to be dropped.")

    def __eq__(self, other):

        type1 = self.type
        range1 = self.rangeval
        nbasis1 = self.nbasis
        pars1 = self.params
        drop1 = self.dropind

        type2 = other.type
        range2 = other.rangeval
        nbasis2 = other.nbasis
        pars2 = other.params
        drop2 = other.dropind

        if type1 != type2:
            return False

        if range1[0] != range2[0] or range1[1] != range2[1]:
            return False

        if nbasis1 != nbasis2:
            return False

        if np.all(drop1 != drop2) != 0:
            return False

        return True

    def __mul__(self, other):
        # Important!!!
        # Currently, this method only support constant and bspline basis
        range1 = self.rangeval
        range2 = other.rangeval

        if range1[0] != range2[0] or range1[1] != range2[1]:
            raise ValueError("Ranges are not equal.")

        # deal with constant bases

        type1 = self.type
        type2 = other.type

        if type1 == 'const' and type2 == 'const':
            prodbasisobj = create_constant_basis(range1)
            return prodbasisobj

        if type1 == 'const':
            return other

        if type2 == 'const':
            return self

        # deal with bspline basis
        # get the number of basis functions
        nbasis1 = self.nbasis
        nbasis2 = other.nbasis
        if type1 == 'bspline' and type2 == 'bspline':

            interiorknots1 = self.params
            interiorknots2 = other.params

            interiorknots12 = np.union1d(interiorknots1, interiorknots2)
            interiorknots12.sort()
            nunique = len(interiorknots12)
            multunique = np.zeros(nunique)

            for i in range(nunique):
                mult1 = interiorknots1 == (interiorknots12[i]) if len(interiorknots1) > 0 else 0
                mult2 = interiorknots2 == (interiorknots12[i]) if len(interiorknots2) > 0 else 0
                multunique[i] = max(sum(mult1), sum(mult2))

            allknots = np.zeros(int(np.sum(multunique)))

            m2 = 0
            for i in range(nunique):
                m1 = m2 + 1
                m2 = int(m2 + multunique[i])
                allknots[m1 - 1:m2] = interiorknots12[i]

            norder1 = nbasis1 - len(interiorknots1)
            norder2 = nbasis2 - len(interiorknots2)
            # norder is not allowed to exceed 20
            norder = min([norder1 + norder2 - 1, 20])
            allbreaks = np.concatenate((np.concatenate((range1[0], allknots), axis=None), range1[1]), axis=None)
            nbasis = len(allbreaks) + norder - 2
            prodbasisobj = create_bspline_basis(rangeval=range1, nbasis=nbasis, norder=norder, breaks=allbreaks)

            return prodbasisobj

        if type1 == 'bspline' or type2 == 'bspline':
            norder = 8
            if type1 == 'bspline':
                interiorknots1 = self.params
                norder1 = nbasis1 - len(interiorknots1)
                norder = min([norder1 + 2, norder])
            if type2 == 'bspline':
                interiorknots2 = other.params
                norder2 = nbasis2 - len(interiorknots2)
                norder = min([norder2 + 2, norder])

            nbasis = max([nbasis1 + nbasis2, norder + 1])
            prodbasisobj = create_bspline_basis(rangeval=range1, nbasis=nbasis, norder=norder)

            return prodbasisobj

    __rmul__ = __mul__


def fourier(x, nbasis, period, nderiv=0):
    # check x and set up range
    n = len(x)
    onen = np.ones(n)
    xrange = [np.min(x), np.max(x)]
    span = xrange[1] - xrange[0]

    # check period and set up omega

    if (period <= 0):
        raise ValueError("Period not positive.")

    omega = 2 * np.pi / period
    omegax = np.inner(omega, x)

    # check nbasis
    if (nbasis <= 0):
        raise ValueError("NBASIS not positive.")

    # check nderiv
    if (nderiv < 0):
        raise ValueError("nderiv is negative.")
    if (nderiv > 0):
        raise ValueError("positive nderiv is not supported currently.")

    # if nbasis is even, add one
    if (nbasis % 2 == 0):
        nbasis = nbasis + 1

    # compute basis matrix
    basismat = np.zeros((n, nbasis))
    if (nderiv == 0):
        basismat[:, 1] = 1 / np.sqrt(2)
        if (nbasis > 1):
            j = np.linspace(2, nbasis - 1, int((nbasis - 1 - 2) / 2) + 1)
            k = np.inner(0.5, j)
            args = np.outer(omegax, k)
            basismat[:, j] = np.sin(args)
            basismat[:, j + 1] = np.cos(args)
    basismat = np.inner(np.sqrt(period / 2), basismat)
    return basismat


# def fourierpen(basisobj, Lfdobj=int2Lfd(0)):
#     if not isinstance(basisobj, basis):
#         raise ValueError("First argument is not a basis object.")
#
#     nbasis = basisobj.nbasis
#     if (nbasis % 2 == 0):
#         basisobj.nbasis = nbasis + 1
#
#     type = basisobj.type
#     if (type != 'fourier'):
#         raise ValueError("Wrong basis type.")
#
#     Lfdobj = int2Lfd(Lfdobj)
#     width = basisobj.rangeval[1] - basisobj.rangeval[0]
#     period = basisobj.params[0]
#     ratio = np.round(width / period)
#     nderiv = Lfdobj.nderiv
#
#     return inprod(basisobj, basisobj, 0, 0)

def rangechk(rangeval):
    nrangeval = len(rangeval)
    OK = True

    if rangeval[0] >= rangeval[1]:
        OK = False
    if nrangeval < 1 or nrangeval > 2:
        OK = False
    return OK


def create_fourier_basis(rangeval=[0, 1], nbasis=3, period=None, dropind=None, quadvals=None, values=None,
                         basisvalues=None):
    type = 'fourier'
    if period is None:
        period = np.diff(rangeval)
    # 1. check RANGEVAL
    if (len(rangeval) < 1):
        if (rangeval <= 0):
            raise ValueError("'len(rangeval) = 0', not allowed.")
        rangeval = [0, rangeval]

    if not rangechk(rangeval):
        raise ValueError("Argument RNAGEVAL is not correct.")

    # Set up the basis object
    basisobj = basis(type=type, rangeval=rangeval, nbasis=nbasis, params=period, dropind=dropind, quadvals=quadvals,
                     values=values, basisvalues=basisvalues)

    return basisobj


def ppBspline(t):
    norder = len(t) - 1
    ncoef = 2 * (norder - 1)
    if norder > 1:
        adds = np.ones(norder - 1)
        tt = np.concatenate((np.concatenate((adds * t[0], t)), adds * t[norder]))
        gapin = np.linspace(1, len(tt) - 1, len(tt) - 1)[np.diff(tt) > 0]
        ngap = len(gapin)
        iseq = [i for i in range(2 - norder, norder)]
        ind = np.outer(np.ones(ngap), iseq) + np.outer(gapin, np.ones(ncoef))
        ind = ind.astype(int)
        tx = np.matrix([tt[i - 1] for i in ind.flatten()]).reshape((ngap, ncoef))
        gapin = gapin.astype(int)
        ty = tx - np.outer([tt[i - 1] for i in gapin], np.ones(ncoef))
        b = np.outer(np.ones(ngap), [i for i in range(1 - norder, 1)]) + np.outer(gapin, np.ones(norder))
        a = np.concatenate((np.concatenate((adds * 0, [1])), adds * 0))
        b = b.astype(int)
        d = np.matrix([a[i - 1] for i in b.flatten()]).reshape((ngap, norder))
        for j in range(1, norder):
            for i in range(1, norder - j + 1):
                ind1 = i + norder - 1
                ind2 = i + j - 1
                d[:, i - 1] = (np.multiply(ty[:, ind1 - 1], d[:, i - 1]) - np.multiply(ty[:, ind2 - 1], d[:, i])) / (
                        ty[:, ind1 - 1] - ty[:, ind2 - 1])
        Coeff = d
        for j in range(2, norder + 1):
            factor = (norder - j + 1) / (j - 1)
            ind = [i for i in range(norder, j - 1, -1)]
            for i in ind:
                Coeff[:, i - 1] = factor * (Coeff[:, i - 1] - Coeff[:, i - 2]) / ty[:, i + norder - j - 1]
        ind = [i for i in range(norder, 0, -1)]
        if ngap > 1:
            Coeff = np.flip(Coeff, 1)
        else:
            Coeff = np.matrix(np.flip(Coeff, 1)).reshape((1, norder))
        index = gapin - norder + 1
    else:
        Coeff = np.matrix(1)
        index = np.matrix(1)
    return [Coeff, index]


def ppderiv(Coeff, Deriv=[0]):
    m = np.shape(Coeff)[0]
    k = np.shape(Coeff)[1]

    if Deriv[0] < 1:
        CoeffD = np.matrix(Coeff)
        return CoeffD

    if k - Deriv[0] < 1:
        CoeffD = np.zeros([m, 1])
        return CoeffD
    else:
        CoeffD = Coeff[:, range(1, k - Deriv[0] + 1)]
        if not isinstance(CoeffD, np.ndarray):
            CoeffD = np.transpose(np.matrix(CoeffD))
            for j in range(1, k - 1):
                bound1 = np.max(1, j - Deriv[0] + 1)
                bound2 = np.min(j, k - Deriv[0])
                CoeffD[:, range(bound1 - 1, bound2)] = (k - j) * CoeffD[:, range(bound1 - 1, bound2)]
        return CoeffD


def polyprod(Coeff1, Coeff2):
    polyorder1 = np.shape(Coeff1)[0]
    norder1 = np.shape(Coeff1)[1]
    polyorder2 = np.shape(Coeff2)[0]
    norder2 = np.shape(Coeff2)[1]
    ndegree1 = polyorder1 - 1
    ndegree2 = polyorder2 - 1

    # if the degrees are not equal, pad out the smaller matrix with 0s
    if ndegree1 != ndegree2:
        if ndegree1 > ndegree2:
            Coeff2 = np.concatenate((Coeff2, np.zeros(ndegree1 - ndegree2, norder2)))
        else:
            Coeff1 = np.concatenate((Coeff1, np.zeros(ndegree2 - ndegree1, norder1)))

    # find order of the product
    D = np.max([ndegree1, ndegree2])
    N = 2 * D + 1

    # compute the coefficients for the products
    convmat = np.zeros([norder1, norder2, N])
    for i in range(D):
        ind = np.array([x + 1 for x in range(i + 1)])
        if len(ind) == 1:
            convmat[:, :, i] = np.outer(Coeff1[ind - 1, :], Coeff2[i - ind + 1, :])
            convmat[:, :, N - i - 1] = np.outer(Coeff1[D - ind + 1, :], Coeff2[D - i + ind - 1, :])
        else:
            convmat[:, :, i] = np.matmul(np.transpose(Coeff1[ind - 1, :]), Coeff2[i - ind + 1, :])
            convmat[:, :, N - i - 1] = np.matmul(np.transpose(Coeff1[D - ind + 1, :]), Coeff2[D - i + ind - 1, :])
    ind = np.array([x + 1 for x in range(D + 1)])
    convmat[:, :, D] = np.matmul(np.transpose(Coeff1[ind - 1, :]), Coeff2[D - ind + 1, :])
    if ndegree1 != ndegree2:
        convmat = convmat[:, :, range(ndegree1 + ndegree2 + 1)]
        convmat = convmat.reshape((norder1, norder2, ndegree1 + ndegree2 + 1))
    return convmat


def inprod_bspline(fdobj1, fdobj2=None, nderiv1=0, nderiv2=0):
    if fdobj2 is None:
        fdobj2 = fdobj1

    if not isinstance(fdobj1, fd):
        raise ValueError("FD1 is not a functional data object.")
    if not isinstance(fdobj2, fd):
        raise ValueError("FD2 is not a functional data object.")

    basis1 = fdobj1.basisobj
    type1 = basis1.type
    if type1 != 'bspline':
        raise ValueError("FDOBJ1 does not have a B-spline basis.")

    range1 = np.array(basis1.rangeval)
    breaks1 = np.concatenate((np.concatenate(([range1[0]], basis1.params)), [range1[1]]))
    nbasis1 = basis1.nbasis
    norder1 = nbasis1 - len(breaks1) + 2

    basis2 = fdobj2.basisobj
    type2 = basis2.type
    if type2 != 'bspline':
        raise ValueError("FDOBJ2 does not have a B-spline basis.")

    range2 = np.array(basis2.rangeval)
    breaks2 = np.concatenate((np.concatenate(([range2[0]], basis2.params)), [range2[1]]))
    nbasis2 = basis2.nbasis
    norder2 = nbasis2 - len(breaks2) + 2

    if np.any(range1 - range2 != 0):
        raise ValueError("The argument ranges from FDOBJ1 and FDOBJ2 are not identical.")

    # check that break values are equal and set up common array
    if len(breaks1) != len(breaks2):
        raise ValueError("The number of knots for FDOBJ1 and FDOBJ2 are not identical.")

    if np.any(breaks1 - breaks2 != 0):
        raise ValueError("The knots of  FDOBJ1 and FDOBJ2 are not identical.")
    else:
        breaks = breaks1

    if len(breaks) < 2:
        raise ValueError("The length of argument BREAKS is less than 2.")

    breakdiff = np.diff(breaks)
    if np.min(breakdiff) <= 0:
        raise ValueError("Argument BREAKS is not strictly increasing.")

    # set up the two coefficient matrixs

    coef1 = np.matrix(fdobj1.coef)
    coef2 = np.matrix(fdobj2.coef)
    if len(np.shape(coef1)) > 2:
        raise ValueError("FDOBJ1 is not univariate.")
    if len(np.shape(coef2)) > 2:
        raise ValueError("FDOBJ2 is not univariate.")

    nbreaks = len(breaks)
    ninterval = nbreaks - 1
    nbasis1 = ninterval + norder1 - 1
    nbasis2 = ninterval + norder2 - 1
    if np.shape(coef1)[0] != nbasis1 or np.shape(coef2)[0] != nbasis2:
        raise ValueError(
            "Error: coef1 should have length no. breaks1 + norder1 - 2, and coef2 no. breaks2 + norder2 - 2.")

    breaks1 = breaks[0]
    breaksn = breaks[nbreaks - 1]

    # the knot sequences are built so that there are no continuity conditions
    # at the first and last breaks. There are k-1 continuity conditions at
    # the other breaks.

    temp = breaks[1: nbreaks - 1]
    knots1 = np.concatenate((np.concatenate((breaks1 * np.ones(norder1), temp)), breaksn * np.ones(norder1)))
    knots2 = np.concatenate((np.concatenate((breaks1 * np.ones(norder2), temp)), breaksn * np.ones(norder2)))

    # construct the piecewise polynomial representations of
    # f^DERIV1 and g^DERIV2

    nrep1 = np.shape(coef1)[1]
    polycoef1 = np.zeros([ninterval, norder1 - nderiv1[0], nrep1])
    for i in range(nbasis1):
        # compute polynomial representation of B(i, norder1, knots1)(x)
        ppBlist = ppBspline(knots1[range(i, i + norder1 + 1)])
        Coeff = ppBlist[0]
        index = ppBlist[1]
        # convert the index of the breaks in knots1 to the index in the variable 'breaks'
        index = index + i - norder1
        CoeffD = ppderiv(Coeff, nderiv1)
        if nrep1 == 1:
            polycoef1[index, :, 1] = coef1[i] * CoeffD + polycoef1[index, :, 1]
        else:
            for j in range(len(index)):
                temp = np.outer(CoeffD[j, :], coef1[i, :])
                polycoef1[index[j], :, :] = temp + polycoef1[index[j], :, :]

    nrep2 = np.shape(coef2)[1]
    polycoef2 = np.zeros([ninterval, norder2 - nderiv2[0], nrep2])
    for i in range(nbasis2):
        # compute polynomial representation of B(i, norder2, knots2)(x)
        ppBlist = ppBspline(knots2[range(i, i + norder2 + 1)])
        Coeff = ppBlist[0]
        index = ppBlist[1]
        # convert the index of the breaks in knots1 to the index in the variable 'breaks'
        index = index + i - norder2
        CoeffD = ppderiv(Coeff, nderiv2)
        if nrep2 == 1:
            polycoef2[index, :, 1] = coef2[i] * CoeffD + polycoef2[index, :, 1]
        else:
            for j in range(len(index)):
                temp = np.outer(CoeffD[j, :], coef2[i, :])
                polycoef2[index[j], :, :] = temp + polycoef2[index[j], :, :]

    # Compute the scalar product between f and g

    prodmat = np.zeros([nrep1, nrep2])
    for j in range(ninterval):
        # multiply f(i1) and g(i2) piecewise and integrate
        c1 = np.matrix(polycoef1[j, :, :])
        c2 = np.matrix(polycoef2[j, :, :])
        polyprodmat = polyprod(c1, c2)
        # compute the coefficients of the anti-derivative
        N = np.shape(polyprodmat)[2]
        delta = breaks[j + 1] - breaks[j]
        power = delta
        prodmati = np.zeros([nrep1, nrep2])
        for i in range(N):
            prodmati = prodmati + power * polyprodmat[:, :, N - i - 1] / (i + 1)
            power = power * delta
        # add the integral to s
        prodmat = prodmat + prodmati

    return prodmat


def eigchk(Cmat):
    if np.max(np.abs(Cmat - np.transpose(Cmat))) / np.max(np.abs(Cmat)) > 1e-10:
        raise ValueError("Cmat is not symmetric.")
    else:
        Cmat = (Cmat + np.transpose(Cmat)) / 2

    # check Cmat  for singularity

    eigval = np.linalg.eigvals(Cmat)
    ncoef = len(eigval)
    if np.min(eigval) <= 0:
        raise ValueError("Non-positive eigenvalue of coefficient matrix.")
    logcondition = np.log10(np.max(eigval)) - np.log10(np.min(eigval))
    if logcondition > 12:
        print("Near singluarity in coefficient matrix.")


def fdchk(fdobj):
    if isinstance(fdobj, fd):
        coef = fdobj.coef
    else:
        if isinstance(fdobj, basis):
            coef = np.zeros([fdobj.nbasis - len(fdobj.dropind), fdobj.nbasis - len(fdobj.dropind)])
            np.fill_diagonal(coef, 1)
            fdobj = fd(coef, fdobj)
        else:
            raise ValueError("FDOBJ is not an FD object.")

    # extract the number of replications and basis object

    coefd = np.shape(np.matrix(coef))
    if len(coefd) > 2:
        raise ValueError("Functional data object must be univariate.")
    nrep = coefd[1]
    basisobj = fdobj.basisobj
    return [nrep, fdobj]


def knotmultchk(basisobj, knotmult):
    type = basisobj.type
    if type == 'bspline':
        params = basisobj.params
        nparams = len(params)
        norder = basisobj.nbasis - nparams
        if norder == 1:
            knotmult = np.concatenate((knotmult, params), axis=None)
        else:
            if nparams > 1:
                for i in range(1, nparams):
                    if params[i] == params[i - 1]:
                        knotmult = np.concatenate((knotmult, params[i]), axis=None)

    return knotmult


def inprod(fdobj1, fdobj2=None, Lfdobj1=int2Lfd(0), Lfdobj2=int2Lfd(0), rng=None, wtfd=0):
    # check FDOBJ1 and get no. replications and basis object
    result1 = fdchk(fdobj1)
    nrep1 = result1[0]
    fdobj1 = result1[1]
    coef1 = fdobj1.coef
    basisobj1 = fdobj1.basisobj
    type1 = basisobj1.type
    range1 = basisobj1.rangeval
    if rng is None:
        rng = range1
    # Default FDOBJ2 to a constant function, using a basis that matches
    # that of FDOBJ1 if possible

    if fdobj2 is None:
        tempfd = fdobj1
        tempbasis = fdobj1.basisobj
        temptype = basisobj1.type
        temprange = basisobj1.rangeval
        if temptype == 'bspline':
            basis2 = create_bspline_basis(temprange, 1, 1)
        else:
            basis2 = create_constant_basis(temprange)
        fdobj2 = fd(1, basis2)

    # check FDOBJ2 and get no. replications and basis object
    result2 = fdchk(fdobj2)
    nrep2 = result2[0]
    fdobj2 = result2[1]
    coef2 = fdobj2.coef
    basisobj2 = fdobj2.basisobj
    type2 = basisobj2.type
    range2 = basisobj2.rangeval

    # check ranges

    if rng[0] < range1[0] or rng[1] > range1[1]:
        raise ValueError("Limits of integration are inadmissible.")

    if isinstance(fdobj1, fd) and isinstance(fdobj2,
                                             fd) and type1 == 'bspline' and type2 == 'bspline' and basisobj1 == basisobj2 and len(
        basisobj1.dropind) == 0 and len(
        basisobj2.dropind) == 0 and wtfd == 0 and np.all(rng == range1):
        inprodmat = inprod_bspline(fdobj1, fdobj2, Lfdobj1.nderiv, Lfdobj2.nderiv)
        return inprodmat

    # check LFDOBJ1 and LFDOBJ2
    Lfdobj1 = int2Lfd(Lfdobj1)
    Lfdobj2 = int2Lfd(Lfdobj2)

    # else proceed with the use of the Romberg integration.

    # set iter
    iter = 0

    # The default case, no multiplicities
    rngvec = rng

    # check for any knot multiplicaties in either argument
    knotmult = []
    if type1 == 'bspline':
        knotmult = knotmultchk(basisobj1, knotmult)
    if type2 == 'bspline':
        knotmult = knotmultchk(basisobj2, knotmult)

    # Modify RNGVEC defining subinverals if there are any
    # knot multiplicities

    if len(knotmult) > 0:
        knotmult = np.unique(knotmult).sort()
        index = np.logical_and(knotmult > rng[0], knotmult < rng[1])
        knotmult = knotmult[index]
        rngvec = np.concatenate((np.concatenate((rng[0], knotmult)), rng[1]))

    # check for either coefficient aray being zero
    if np.all(coef1 == 0) or np.all(coef2 == 0):
        return np.zeros([nrep1, nrep2])

    ##################################
    # loop through sub-intervals
    ##################################

    JMAX = 15
    JMIN = 5
    EPS = 1e-4

    inprodmat = np.zeros([nrep1, nrep2])
    nrng = len(rngvec)
    for irng in range(1, nrng):
        rngi = [rngvec[irng - 1], rngvec[irng]]
        # change range so as to avoid being exactly one
        # multiple knot values
        if irng > 2:
            rngi[0] = rngi[0] + 1e-10
        if irng < nrng:
            rngi[1] = rngi[1] - 1e-10

        # set up first iteration

        iter = 1
        width = rngi[1] - rngi[0]
        JMAXP = JMAX + 1
        h = np.ones(JMAXP)
        h[1] = 0.25
        s = np.zeros([JMAXP, nrep1, nrep2])
        sdim = len(np.shape(s))
        # the first iterations uses just the endpoints
        fx1 = eval_fd(rngi, fdobj1, Lfdobj1)
        fx2 = eval_fd(rngi, fdobj2, Lfdobj2)
        # multiply by values of weight function if necessary
        # if isinstance(wtfd, fd):
        #     wtd = eval_fd(rngi, wtfd, 0)
        #     fx2 = np.multiply(wtd.reshape((np.shape(wtd)[0], np.shape(fx2)[1])), fx2)
        s[0, :, :] = width * np.matmul(np.transpose(fx1), fx2).reshape((nrep1, nrep2)) / 2
        tnm = 0.5

        # now iterate to convergence
        for iter in range(2, JMAX):
            tnm = tnm * 2
            if iter == 2:
                x = np.mean(rngi)
            else:
                dele = width / tnm
                x = np.linspace(rngi[0] + dele / 2, rngi[1] - dele / 2, int((rngi[1] - rngi[0]) / dele))
            fx1 = eval_fd(x, fdobj1, Lfdobj1)
            fx2 = eval_fd(x, fdobj2, Lfdobj2)
            if not isinstance(wtfd, int):
                wtd = eval_fd(wtfd, x, 0)
                fx2 = np.multiply(np.repeat(wtd, np.shape(fx2)[1]).reshape((np.shape(wtd)[0], np.shape(fx2)[1])), fx2)
            chs = width * np.matmul(np.transpose(fx1), fx2).reshape((nrep1, nrep2)) / tnm
            s[iter - 1, :, :] = (s[iter - 2, :, :] + chs) / 2
            if iter >= 5:
                ind = range(iter - 5, iter)
                ya = s[ind, :, :]
                ya = ya.reshape((5, nrep1, nrep2))
                xa = h[ind]
                absxa = np.abs(xa)
                absxamin = np.min(absxa)
                ns = int(np.min(np.linspace(0, len(absxa) - 1, len(absxa))[absxa == absxamin]))
                cs = ya
                ds = ya
                y = ya[ns, :, :]
                ns = ns - 1
                for m in range(4):
                    for i in range(4 - m):
                        ho = xa[i]
                        hp = xa[i + m]
                        w = (cs[i + 1, :, :] - ds[i, :, :]) / ((ho - hp) if (ho - hp) > 1e-6 else 1e-6)
                        ds[i, :, :] = hp * w
                        cs[i, :, :] = ho * w
                    if 2 * ns < 4 - m:
                        dy = cs[ns, :, :]
                    else:
                        dy = ds[ns - 1, :, :]
                        ns = ns - 1
                    y = y + dy
                ss = y
                errval = np.max(np.abs(dy))
                ssqval = np.max(np.abs(ss))
                if np.all(ssqval > 0):
                    crit = errval / ssqval
                else:
                    crit = errval
                if crit < EPS and iter >= JMIN:
                    break
            s[iter, :, :] = s[iter - 1, :, :]
            h[iter] = 0.25 * h[iter - 1]
        inprodmat = inprodmat + ss

    if len(np.shape(inprodmat)) == 2:
        return np.matrix(inprodmat)
    else:
        return inprodmat


class fRegressArg:
    def __init__(self, yfd, xfdlist, betalist, wt):
        self.yfd = yfd
        self.xfdlist = xfdlist
        self.betalist = betalist
        self.wt = wt


class fRegressList:
    def __init__(self, yfdobj, xfdlist, betalist, betaestlist, yhatfdobj, Cmat, Dmat, Cmatinv, wt, df, y2cMap, SigmaE,
                 betastderrlist, bvar, c2bMap):
        self.yfdobj = yfdobj
        self.xfdlist = xfdlist
        self.betalist = betalist
        self.betaestlist = betaestlist
        self.yhatfdobj = yhatfdobj
        self.Cmat = Cmat
        self.Dmat = Dmat
        self.Cmatinv = Cmatinv
        self.wt = wt
        self.df = df
        self.y2cMap = y2cMap
        self.SigmaE = SigmaE
        self.betastderrlist = betastderrlist
        self.bvar = bvar
        self.c2bMap = c2bMap


def fRegressArgCheck(yfd, xfdlist, betalist, wt=None):
    if not (isinstance(yfd, fdPar) or isinstance(yfd, fd)):
        raise ValueError("First argument is not of class 'fdPar' and 'fd'.")

    if isinstance(yfd, fdPar):
        yfd = yfd.fd

    if isinstance(yfd, fd):
        ycoef = yfd.coef
        N = np.shape(ycoef)[1]
    else:
        N = len(yfd)

    # check that xfdlist is a list object and compute number of covaraites p

    # check XFDLIST

    if isinstance(xfdlist, fd):
        xfdlist = [xfdlist]

    # get number of independent variables p

    p = len(xfdlist)

    # check BETALIST

    if isinstance(betalist, fd):
        betalist = [betalist]

    if len(betalist) != p:
        raise ValueError("Number of regression coefficients does not match number of independent variables.")

    # extract the range if YFD is functional

    if isinstance(yfd, fd):
        rangeval = yfd.basisobj.rangeval
    else:
        rangeval = [0, 1]

    # check contents of XFDLIST

    onebasis = create_constant_basis(rangeval)
    onesfd = fd(1, onebasis)

    xerror = False

    for j in range(p):
        xfdj = xfdlist[j]
        if isinstance(xfdj, fd):
            xcoef = xfdj.coef
            if len(np.shape(xcoef)) > 2:
                raise ValueError("Covariate ", j, " is not univariate.")

            Nj = np.shape(xcoef)[1]
            if Nj != N:
                print("Incorrect number of replications in XFDLIST for covariate ", j)
                xerror = True
        elif isinstance(xfdj, np.ndarray):
            xfdj = np.matrix(xfdj)
            Zdimj = np.shape(xfdj)
            if Zdimj[0] != N:
                print("vector in XFDLIST has wrong length.")
                xerror = True
            if Zdimj[1] != 1:
                print("vector in XFDLIST has wrong length.")
                xerror = True
            xfdlist[j] = fd(np.matrix(xfdj).reshape((1, N)), onebasis)

    # check contents of BETALIST

    berror = False
    for j in range(p):
        betafdParj = betalist[j]
        if isinstance(betafdParj, fd) or isinstance(betafdParj, basis):
            betafdParj = fdPar(betafdParj)
            betalist[j] = betafdParj
        if not isinstance(betafdParj, fdPar):
            print("BETALIST  is not  a fdPar object.")
            berror = True

    if xerror or berror:
        raise ValueError("An error has been found in either XFDLIST or BETALIST.")

    if wt is None:
        wt = np.ones(N)

    return fRegressArg(yfd, xfdlist, betalist, wt)


def fRegress(y, xfdlist, betalist, wt=None, y2cMap=None, SigmaE=None, returnMatrix=False, method='fRegress', sep='.'):
    arglist = fRegressArgCheck(y, xfdlist, betalist, wt)
    yfdobj = arglist.yfd
    xfdlist = arglist.xfdlist
    betalist = arglist.betalist
    wt = arglist.wt
    p = len(xfdlist)
    wtconst = np.var(wt) == 0
    ycoef = yfdobj.coef
    ycoefdim = np.shape(ycoef)
    N = ycoefdim[1]
    ybasisobj = yfdobj.basisobj
    rangeval = ybasisobj.rangeval
    ynbasis = ybasisobj.nbasis
    onesbasis = create_constant_basis(rangeval)
    onesfd = fd(1, onesbasis)
    if len(ycoefdim) > 2:
        raise ValueError("YFDOBJ from YFD is not univariate.")
    ncoef = 0
    for j in range(p):
        betafdParj = betalist[j]
        if betafdParj.estimate:
            ncoefj = betafdParj.fd.basisobj.nbasis
            ncoef = ncoef + ncoefj
    Cmat = np.zeros([ncoef, ncoef])
    Dmat = np.zeros(ncoef)
    mj2 = 0
    # j = 1
    # xfdj = xfdlist[j]
    # xyfdj = xfdj * yfdobj
    # wtfdj = xyfdj.sum()
    # betafdj = betalist[j].fd
    # betabasisj = betafdj.basis
    # ncoefj = betabasisj.nbasis
    # k =3
    # betafdk = betalist[k].fd
    # betabasisk = betafdk.basis
    # ncoefk = betabasisk.nbasis
    for j in range(p):
        betafdParj = betalist[j]
        if betafdParj.estimate:
            betafdj = betafdParj.fd
            betabasisj = betafdj.basisobj
            ncoefj = betabasisj.nbasis
            mj1 = mj2
            mj2 = mj1 + ncoefj
            xfdj = xfdlist[j]
            if wtconst:
                xyfdj = xfdj * yfdobj
            else:
                xyfdj = xfdj * wt * yfdobj
            wtfdj = xyfdj.sum()
            Dmatj = inprod(betabasisj, onesfd, 0, 0, rangeval, wtfdj)
            Dmat[mj1: mj2] = Dmatj.reshape(mj2 - mj1)
            mk2 = 0
            for k in range(j + 1):
                betafdPark = betalist[k]
                if betafdPark.estimate:
                    betafdk = betafdPark.fd
                    betabasisk = betafdk.basisobj
                    ncoefk = betabasisk.nbasis
                    mk1 = mk2
                    mk2 = mk1 + ncoefk
                    xfdk = xfdlist[k]
                    if wtconst:
                        xxfdjk = xfdj * xfdk
                    else:
                        xxfdjk = xfdj * wt * xfdk
                    wtfdjk = xxfdjk.sum()
                    Cmatjk = inprod(betabasisj, betabasisk, 0, 0, rangeval, wtfdjk)
                    Cmat[mj1: mj2, mk1:mk2] = Cmatjk
                    Cmat[mk1:mk2, mj1: mj2] = np.transpose(Cmatjk)
            lamdbaj = betafdParj.lamdba
            if lamdbaj > 0:
                Rmatj = betafdParj.penmat
                if Rmatj is None:
                    Lfdj = betafdParj.Lfd
                    Rmatj = eval_penalty(betabasisj, Lfdj)
                Cmat[mj1: mj2, mj1: mj2] = Cmat[mj1: mj2, mj1: mj2] + lamdbaj * Rmatj
    Cmat = (Cmat + np.transpose(Cmat)) / 2

    eigchk(Cmat)
    Cmatinv = np.linalg.inv(Cmat)
    # Lmat = np.linalg.cholesky(Cmat)
    # Lmatinv = np.linalg.inv(Lmat)
    # Cmatinv = np.matmul(Lmatinv, np.transpose(Lmatinv))
    betacoef = np.dot(Cmatinv, Dmat)
    betaestlist = copy.deepcopy(betalist)
    mj2 = 0
    for j in range(p):
        betafdParj = copy.deepcopy(betalist[j])
        if betafdParj.estimate:
            betafdj = betafdParj.fd
            ncoefj = betafdj.basisobj.nbasis
            mj1 = mj2
            mj2 = mj2 + ncoefj
            coefj = betacoef[mj1: mj2]
            betafdj.coef = np.transpose(np.matrix(coefj))
            betafdParj.fd = betafdj
        betaestlist[j] = betafdParj
    nfine = max(501, 10 * ynbasis + 1)
    tfine = np.linspace(rangeval[0], rangeval[1], nfine)
    yhatmat = np.zeros([nfine, N])
    for j in range(p):
        xfdj = xfdlist[j]
        xmatj = eval_fd(tfine, xfdj, 0, returnMatrix)
        betafdParj = betaestlist[j]
        betafdj = betafdParj.fd
        betavecj = eval_fd(tfine, betafdj, 0, returnMatrix)
        for k in range(np.shape(xmatj)[1]):
            xmatj[:, k] = np.multiply(xmatj[:, k], np.matrix(betavecj))
        yhatmat = yhatmat + xmatj
    yhatfdobj = smooth_basis(tfine, yhatmat, ybasisobj).fd
    df = None
    if not (y2cMap is None or SigmaE is None):
        y2cdim = np.shape(y2cMap)
        if y2cdim[0] != ynbasis or y2cdim[1] != np.shape(SigmaE)[0]:
            raise ValueError("Dimensions of Y2CMAP not correct.")
        ybasismat = eval_basis(tfine, ncoef, ynbasis * N)
        deltat = tfine[1] - tfine[0]
        basisprodmat = np.zeros([ncoef, ynbasis * N])
        mj2 = 0
        for j in range(p):
            betafdParj = betalist[j]
            betabasisj = betafdParj.fd.basisobj
            ncoefj = betabasisj.nbasis
            bbasismatj = eval_basis(tfine, betabasisj, 0, returnMatrix)
            xfdj = xfdlist[j]
            tempj = eval_fd(tfine, xfdj, 0, returnMatrix)
            mj1 = mj2 + 1
            mj2 = mj2 + ncoefj
            mk2 = 0
            for k in range(ynbasis):
                mk1 = mk2 + 1
                mk2 = mk2 + N
                tempk = bbasismatj * ybasismat[:, k]
                basisprodmat[mj1 - 1: mj2, mk1 - 1: mk2] = deltat * np.matmul(np.transpose(tempk), tempj)
        c2bMap = np.linalg.solve(Cmat, basisprodmat)
        VarCoef = np.dot(np.dot(y2cMap, SigmaE), np.transpose(y2cMap))
        CVariance = np.kron(VarCoef, np.identity(N))
        bvar = np.dot(np.dot(c2bMap, CVariance), np.transpose(c2bMap))
        betastderrlist = []
        mj2 = 0
        for j in range(p):
            betafdParj = betalist[j]
            betabasisj = betafdParj.fd.basisobj
            ncoefj = betabasisj.nbasis
            mj1 = mj2 + 1
            mj2 = mj2 + ncoefj
            bbasismat = eval_basis(tfine, betabasisj, 0, returnMatrix)
            bvarj = bvar[mj1 - 1: mj2, mj1 - 1: mj2]
            bstderrj = np.sqrt(np.dot(np.dot(bbasismat, bvarj), np.transpose(bbasismat)).diagonal())
            bstderrfdj = smooth_basis(tfine, bstderrj, betabasisj).fd
            betastderrlist.append(bstderrfdj)
    else:
        betastderrlist = None
        bvar = None
        c2bMap = None

    fr = fRegressList(yfdobj, xfdlist, betalist, betaestlist, yhatfdobj, Cmat, Dmat, Cmatinv, wt, df, y2cMap, SigmaE,
                      betastderrlist, bvar, c2bMap)
    return fr



def setGenerator(set, samplesPerWorker, numworkers):
    res = []
    for i in set:
        res.extend([i + samplesPerWorker * n for n in range(numworkers)])
    return res


