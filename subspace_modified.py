# -*- coding: utf-8 -*-
# Author: Konstantinos

import os
import sys
from abc import ABCMeta
import matplotlib.pyplot as plt
import scipy.linalg as splinalg
import numpy as np
import scipy as sp

__author__ = 'Konstantinos Tatsis'
__email__ = 'konnos.tatsis@gmail.com'

# Contains:

# 1. Deterministic Subspace Identification (DSI)
# 2. Stochastic Subspace Identification (SSI)
# 3. Combined Stochastic Deterministic Subspace (CSI)   To be done
# 4. Eigensystem realization (ERA)


def addNoise(signal, level):

    """
    Add Gaussian white noise to signals.

    Parameters
    ----------
    signal: ndarray
        An (n x s) array containing the signals to be corrupted, where n is
        the number of samples and s is the number of signals.
        n is the 2* sensors, s is the time series
    level: float, positive
        The level of noise to be added, as percentage of the signal standard
        deviation.
       
    Returns
    -------
    signal: ndarray
        An array containing the noisy signals.
    """
    
    # try:
    #     signal.shape[1]
    # except IndexError:
    #     signal = np.array([signal])


    # allow for 2-dimensional level, for each signal separately.
    
    mean = np.mean(signal, axis=0)
    std = np.sqrt(np.sum((signal-mean)**2/signal.shape[0], axis=0))
    
    #print(std)

    noisy_signal = signal+level*std*np.random.normal(size=signal.shape)
    noisy_signal = noisy_signal.squeeze()
    #print(noisy_signal)
    
    return noisy_signal


def MAC(u, v):

    """
    Modal Assurance Criterion between vectors u and v.

    Parameters
    ----------
    u: ndarray
        An 1d or 2d array
    v: ndarray
        An 1d or 2d array

    Returns
    -------
    MAC: ndarray
        If u and v are 1-dimensional arrays it contains their MAC
        value. If u and v are 2-dimensional arrays it contains 
        their MAC matrix.
    """
    
    try:
        u.shape[1]
    except IndexError:
        u = np.array([u]).T

    try:
        v.shape[1]
    except IndexError:
        v = np.array([v]).T

    # check also if u and v are of same size in the first axis

    MAC = np.zeros((u.shape[1], v.shape[1]), dtype=complex)

    for i, a in enumerate(u.T):
        for j, b in enumerate(v.T):
            nom = (a.conj().dot(b))**2
            den = (a.conj().dot(a))*(b.conj().dot(b))
            MAC[i, j] = nom/den
    
    return MAC
    


#   Need to code the following methods:

#   1. Deterministic subspace identification (DSI)
#   2. Combine Subspace Identification (N4SID, CVA MOESP)



class SI:

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def plotSingularValues(self):
        
        plt.figure('Singular values - '\
                   'Stochastic Subspace Identification (SSI)',
                    figsize=(10, 3.5))
        plt.title('Singular values', fontsize=12)
        plt.grid()

        plt.plot(np.arange(1, self.__S.shape[0]+1),self.__S,
                 '.', markersize=6, color=(1, 0, 0))
        plt.yscale('log')

        plt.xlabel('Number', fontsize=11)
        plt.ylabel('Value [-]', fontsize=11)

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        plt.show()


    def plotStabilizationDiagram(self):
        pass





        # Up = U[:n_out*rows, :]
        # Uf = U[n_out*rows:, :]

        # Up_ = U[:n_out*(rows+1), :]
        # Uf_ = U[n_out*(rows+1):, :]

        # Yp = Y[:n_out*rows, :]
        # Yf = Y[n_out*rows:, :]

        # Yp_ = Y[:n_out*(rows+1), :]
        # Yf_ = Y[n_out*(rows+1):, :]

        # Wp = np.vstack((Up, Yp))
        # Wp_ = np.vstack((Up_, Yp_))





class DSI(object):

    """

    Deterministic Subspace Identification algorithm. Computes the 
    state-space model of purely deterministic systems, with no measurement
    nor process noise, from given input-output data.


    Attributes:
    ----------

    Methods:
    ----------
    getSingularValues()
        ...
    plotSingularValues()
        ...
    getStablePoles()
        ...
    plotStablePoles()
        ...
    getStabilizationDiagram()
        ...
    getRealization(order)
        ...

    """

    def __init__(self, input_, output, sPeriod, rows, columns=None):

        """
        Parameters:
        ----------
        input_: ndarray
            An array containing the data samples of the measured inputs.
        output: ndarray
            An array containing the data samples of the observed outputs.
        sPeriod: real
            The sampling period of the input and output vectors.
        rows: int
            The number of block rows of the input and output block
            Hankel matrices. It should at least be larger than the
            maximum order of the system one wants to identify.
        columns: int, optional
            The number of columns of the input and output block 
            Hankel matrices. When the number of columns is not
            specified, all given data samples are used for the
            construction of the block Hankel matrices.

        Raises:
        ----------
        IndexError:
        """

        # 1. rows is not necessary to be stored as an attribute

        try:
            input_.shape[1]
        except IndexError:
            input_ = input_.reshape((1, input_.shape[0]))

        try:
            output.shape[1]
        except IndexError:
            output = output.reshape((1, output.shape[0]))

        self.input = input_
        self.output = output
        self.samp_per = sPeriod
        self.rows = rows

        n_in = int(self.input.shape[0])
        n_out = int(self.output.shape[0])
        samples = int(self.output.shape[1])

        if columns==None or columns>samples-2*rows+1:
            columns = samples-2*rows+1

        denom = np.sqrt(columns)

        U = np.zeros((2*rows*n_in, columns))

        for row in range(2*rows):
            for column, j in enumerate(range(row, row+columns)):
                U[row*n_in:(row+1)*n_in, column] = self.input[:, j]/denom

        Y = np.zeros((2*rows*n_out, columns))

        for row in range(2*rows):
            for column, j in enumerate(range(row, row+columns)):
                Y[row*n_out:(row+1)*n_out, column] = self.output[:, j]/denom


        W = np.vstack((U, Y))
        Q, R = np.linalg.qr(W.T)
        L, Q = R.T, Q. T

        self.__Q = Q
        self.__L = L

        # calculate weight matrices W1 and W2
        algorithm = 'N4SID'

        if algorithm=='N4SID':
            W1 = sp.sparse.eye(n_out*rows)
            W2 = sp.sparse.eye(columns)
            # W1 = np.eye(n_out*rows)
            # W2 = np.eye(columns)
        elif algorithm=='CVA':
            W1 = 0
            W2 = 0
        elif algorithm=='MOESP':
            W1 = np.eye(n_out*rows)
            W2 = 0
        else:
            raise TypeError('Invalid type of algorithm')

        self.__W1 = W1
        self.__W2 = W2


        indexA = 2*n_in*rows+n_out*rows
        indexB = n_in*rows

        Lleft = L[:indexA, :][:, :indexA]
        Lright = L[indexA:, :][:, :indexA]
        LUY = Lright.dot(np.linalg.pinv(Lleft)) # Make a check if Lleft is full rank
        LUp, LYp = LUY[:, :indexB], LUY[:, 2*indexB:]

        contribution1 = LUp.dot(L[:indexB, :][:, :indexA])
        contribution2 = LYp.dot(L[2*indexB:indexA, :][:, :indexA])

        obliqueProjection = (contribution1+contribution2).dot(Q[:indexA, :])

        # obliqueProjection = W1.dot(obliqueProjection).dot(W2)

        U, S, V = np.linalg.svd(obliqueProjection)

        self.__U = U
        self.__S = S
        self.__V = V


    def getSingularValues(self):
        return self.__S


    def plotSingularValues(self):
        plt.figure('Deterministic Subspace Identification (DSI) - '\
                   'Singular values', figsize=(10, 3.5))
        plt.title('Singular values', fontsize=12)
        plt.grid()

        plt.plot(np.arange(1, self.__S.shape[0]+1),self.__S,
                 '.', markersize=6, color=(1, 0, 0))
        plt.yscale('log')

        plt.xlabel('Number', fontsize=11)
        plt.ylabel('Value [-]', fontsize=11)

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        plt.show()


    def getStablePoles(self, minOrder=2, maxOrder=20,
                        frequencyTol=1e-2, dampingTol=5e-2, modeTol=1e-2,
                        minFrequency=0, maxFrequency=1e3,
                        minDamping=1e-1, maxDamping=1e1):
        pass


    def plotStablePoles(self):#self, minOrder=2, maxOrder=20,
                        # frequencyTol=1e-2, dampingTol=5e-2, modeTol=1e-2,
                        # minFrequency=0, maxFrequency=1e3,
                        # minDamping=1e-1, maxDamping=1e1):

        plt.figure('Deterministic Subspace Identification algorithm (DSI) - '\
                   'Stabilization diagram', figsize=(10,5))
        plt.title('Stabilization diagram', fontsize=12)

        plt.plot(frequencies[0], frequencies[1],
                 'o', markersize=4, fillstyle='none',
                 label='Stable frequency')
        plt.plot(dampings[0], dampings[1],
                 'x', markersize=5, fillstyle='none',
                 label='Stable damping ratio')
        plt.plot(modes[0], modes[1],
                 's', markersize=5, fillstyle='none',
                 label='Stable mode shape')

        plt.xlabel('Frequency [Hz]', fontsize=11)
        plt.ylabel('Model order', fontsize=11)

        plt.xlim([0, maxFrequency])
        plt.ylim([minOrder, maxOrder])

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()


    def getStabilizationDiagram(self, minOrder=2, maxOrder=20, 
                            frequencyTol=1e-2, dampingTol=5e-2, modeTol=1e-2,
                            minFrequency=0, maxFrequency=1e3,
                            minDamping=1e-1, maxDamping=1e1):

        if minOrder%2 != 0:
            raise ValueError('Minimum order must be an even number larger \
                             than or equal to 2')

        if maxOrder%2 != 0 or maxOrder-minOrder < 2:
            raise ValueError('Maximum order must be an even number larger \
                             than the minimum order')

        n_out = int(self.output.shape[0])
        n_in = int(self.input.shape[0])

        rows_in = self.rows_in
        rows_out = self.rows_out

        W1 = self.__W1
        L = self.__L
        Q = self.__Q

        Ui = 0 ### !!!!
        Yi = 0 ### !!!!

        properties = []

        for order in range(minOrder, maxOrder+2, 2):\

            # Implement Algorithm 2

            U1 = self.__U[:, :order]
            S1 = self.__S[:order]
            V1 = self.__V[:order, :]
            U2 = self.__U[:, order:]

            sqrtS1 = np.diag(np.sqrt(S1))

            # Ob_proj = U1.dot(np.diag(S1)).dot(V1)
            # Ob_proj_ = 0 ###  !!!!!!

            # Oi = np.linalg.inv(W1).dot(U1).dot(sqrtS1)
            # Oi = U2.T.dot(W1)

            # These are for the first algorithm
            # Oi = U1.dot(sqrtS1)
            # Oi_ = Oi[:int(Oi.shape[0])-n_out, :]

            Oi = U1.dot(sqrtS1)
            # Oip = U2.T

            Oit = Oi[:int(Oi.shape[0])-n_out, :]
            Oib = Oi[n_out:, :]

            A = np.linalg.pinv(Oit).dot(Oib)
            C = Oi[:n_out, :]


            # These are for the first algorithm
            # Xi = np.linalg.pinv(Oi).dot(Ob_proj)
            # Xi_ = np.linalg.pinv(Oi_).dot(Ob_proj_)

            # ABCD = np.vstack((Xi_, Yi)).dot(
            #        np.linalg.pinv(np.vstack((Xi, Ui))))

            # A = ABCD[:order, :order]
            # B = ABCD[:order, order:]
            # C = ABCD[order:, :order]
            # D = ABCD[order:, order:]

            lamda, psi = np.linalg.eig(A)
            index = np.where(np.imag(lamda) >= 0)
            lamda, psi = lamda[index[0]], psi[:, index[0]]
            lamda = np.log(lamda)/ self.samp_per

            properties.append([order,
                               np.abs(lamda)/(2*np.pi),
                               -100*np.real(lamda)/np.abs(lamda),
                               C.dot(psi)])

        frequencies = [[], []]
        dampings = [[], [], []]
        modes = [[], [], []]

        for p in range(1, len(properties)):
            order = properties[p][0]
            for q in range(len(properties[p][1])):
                freq1 = properties[p][1][q]
                damp1 = properties[p][2][q]
                mode1 = properties[p][3][:, q]
                
                freq_condition = freq1 < minFrequency or freq1 > maxFrequency
                damp_condition = damp1 < minDamping or damp1 > maxDamping

                if freq_condition or damp_condition:
                    continue

                for r in range(len(properties[p-1][1])):
                    freq0 = properties[p-1][1][r]
                    damp0 = properties[p-1][2][r]
                    mode0 = properties[p-1][3][:, r]
                    if np.abs((freq0-freq1)/freq0) <= frequencyTol:
                        frequencies[0].append(freq1)
                        frequencies[1].append(order)
                    if np.abs((damp0-damp1)/damp0) <= dampingTol:
                        dampings[0].append(freq1)
                        dampings[1].append(order)
                        dampings[2].append(damp1)
                    if (1-MAC(mode0, mode1)) < modeTol:
                        modes[0].append(freq1)
                        modes[1].append(order)
                        modes[2].append(mode1)

        plt.figure('Deterministic Subspace Identification algorithm (DSI) - '\
                   'Stabilization diagram', figsize=(10,5))
        plt.title('Stabilization diagram', fontsize=12)

        plt.plot(frequencies[0], frequencies[1],
                 'o', markersize=4, fillstyle='none',
                 label='Stable frequency')
        plt.plot(dampings[0], dampings[1],
                 'x', markersize=5, fillstyle='none',
                 label='Stable damping ratio')
        plt.plot(modes[0], modes[1],
                 's', markersize=5, fillstyle='none',
                 label='Stable mode shape')

        plt.xlabel('Frequency [Hz]', fontsize=11)
        plt.ylabel('Model order', fontsize=11)

        plt.xlim([0, maxFrequency])
        plt.ylim([minOrder, maxOrder])

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()


    def getRealization(self, order):
        pass




class CSI(object):

    """
        ...
    """

    def __init__(self, input_, output, samPer, rows, columns=None):
        
        try:
            input_.shape[1]
        except IndexError:
            input_ = input_.reshape((1, input_.shape[0]))

        try:
            output.shape[1]
        except IndexError:
            output = output.reshape((1, output.shape[0]))

        if input_.shape[1]!=output.shape[1]:
            raise TypeError('')

        self.input = input_
        self.output = output
        self.samPer = samPer
        self.rows = rows

        n_in = int(self.input.shape[0])
        n_out = int(self.output.shape[0])
        samples = int(self.output.shape[1])

        if columns==None or columns>samples-2*rows+1:
            columns = samples-2*rows+1

        denom = np.sqrt(columns)

        U = np.zeros((2*rows*n_in, columns))

        for row in range(2*rows):
            for column, j in enumerate(range(row, row+columns)):
                U[row*n_in:(row+1)*n_in, column] = self.input[:, j]/denom

        Y = np.zeros((2*rows*n_out, columns))

        for row in range(2*rows):
            for column, j in enumerate(range(row, row+columns)):
                Y[row*n_out:(row+1)*n_out, column] = self.output[:, j]/denom

        W = np.vstack((U, Y))
        Q, R = np.linalg.qr(W.T)
        L, Q = R.T, Q.T
        self.__Q, self.__L = Q, L





class SSI:

    # Modifications for cov/ref and data/ref.

    """
    Stochastic Subspace Identification (SSI-Cov and SSI-Data) for purely 
    stochastic systems with no external input.

    Parameters
    ----------
    output: ndarray
        The measured system outputs, which has as many rows as the system 
        outputs and as many columns as the time samples.
    speriod: float
        The sampling period of measured outputs.
    rows: int
        The number of block rows for the output block Hankel matrix.
    columns: None
        The number of block columns for the output block Hankel matrix.
    method: {'cov', 'data'}
        The method to be used, with 'cov' and 'data' representing the 
        covariance- and data-driven methods respectively.
    weight: {'PC', 'UPC', 'CVA'}, optional
        The weighting scheme
            'PC'    Principal component
            'UPC'   Unweighted principal component
            'CVA'   Canonical variate analysis

    Methods
    -------

    Raises
    ------
    ValueError
        If a non-positive sampling period is specified.
        If a negative or non-integer number of rows or columns is specified.
        If an invalid method or weight algorithm is specified.
    """

    def __init__(self, output, speriod, rows, columns=None, method='cov', weight='PC'):

        try:
            output.shape[1]
        except IndexError:
            output = output.reshape((1, output.shape[0]))

        if speriod <= 0:
            raise ValueError('Non-positive sampling period.')


        if not isinstance(rows, int):
            raise TypeError('Number of rows must be positive integer.')

        if rows <= 0:
            raise ValueError('Negative number of rows.')


        if columns is not None:
            if not isinstance(columns, int):
                raise TypeError('Number of columns must be positive integer.')

            if columns <= 0:
                raise ValueError('Non-positive number of columns.')


        if method not in ['cov', 'data']:
            raise ValueError('Invalid method.')

        if weight not in ['PC', 'UPC', 'CVA']:
            raise ValueError('Invalid weight algorithm.')

        header = 'Stochastic Subspace Identification (SSI) - {}\n'
        sys.stdout.write(header.format(method.capitalize()))

        samples = output.shape[1]
        channels = output.shape[0]

        if columns is None or columns > samples-rows-1:
            columns = samples-2*rows+1

        self.channels = channels
        self.speriod = speriod
        self.method = method

        try:
            denom = np.sqrt(columns)
            Y = np.zeros((2*rows*channels, columns))
        except MemoryError:
            error = 'Reduce number of samples or number of channels.'
            raise MemoryError(error)

        # Construct block-Hankel matrix

        sys.stdout.write('  forming block Hankel matrix...\n')

        for row in range(2*rows):
            for column, j in enumerate(range(row, row+columns)):
                Y[row*channels: (row+1)*channels, column] = output[:, j]/denom


        Yp = Y[:rows*channels, :]
        Yf = Y[rows*channels:, :]

        if method is 'cov':
            sys.stdout.write('  performing singular value decomposition...\n')
            T1i = Yf.dot(Yp.T)
            U, S, V = np.linalg.svd(T1i)
        else:
            sys.stdout.write('  performing QR decomposition...\n')
            Q, R = np.linalg.qr(Y.T)
            Q, R = Q.T, R.T
            self.__Q = Q
            self.__R = R

            sys.stdout.write('  calculating weight matrices...\n')
            if weight == 'PC':
                # Use of QR decomposition for W2
                R11 = R[:rows*channels, :rows*channels]
                YpYp = R11.dot(R11.T)
                W1 = np.eye(rows*channels)
                W2 = Yp.T.dot(np.linalg.inv(sp.linalg.sqrtm(YpYp))).dot(Yp)
            elif weight == 'UPC':
                W1 = np.eye(rows*channels)
                W2 = np.eye(columns)
            elif weight == 'CVA':
                # Use of QR decomposition for W1
                W1 = np.linalg.inv(sp.linalg.sqrtm(Yf.dot(Yf.T)))
                W2 = np.eye(columns)

            # Calculate oblique projection
            Op = R[rows*channels:, :rows*channels].dot(Q[:rows*channels, :])
            Op = W1.dot(Op).dot(W2)

            sys.stdout.write('  performing singular value decomposition...\n')
            U, S, V = np.linalg.svd(Op)

        sys.stdout.write('  processing completed.\n')
        tol = 1e-10

        if self.method == 'cov':

            def getProperties(U1, S1, V1):
                sqS1 = np.diag(np.sqrt(S1))
                Oi = U1.dot(sqS1)
                Ci = sqS1.dot(V1)
                Oit = Oi[:Oi.shape[0]-self.channels, :]
                Oib = Oi[self.channels:, :]
                A = np.linalg.pinv(Oit).dot(Oib)
                C = Oi[:self.channels, :]

                lamda, psi = np.linalg.eig(A)

                index = np.where(np.imag(lamda) >= 0)
                lamda, psi = lamda[index[0]], psi[:, index[0]]
                index = np.where(np.abs(lamda) >= tol)
                lamda, psi = lamda[index[0]], psi[:, index[0]]

                lamda = np.log(lamda)/self.speriod

                frequencies = np.abs(lamda)/(2*np.pi)
                ratios = -100*np.real(lamda)/np.abs(lamda)
                shapes = C.dot(psi)

                indices = np.argsort(frequencies)
                frequencies = frequencies[indices]
                ratios = ratios[indices]
                shapes = shapes[:, indices]

                return frequencies, ratios, shapes

        else:

            def getProperties(U1, S1, V1):
                sqS1 = np.diag(np.sqrt(S1))
                Pi = U1.dot(np.diag(S1)).dot(V1)
                Oi = U1.dot(sqS1)
                Xi = np.linalg.pinv(Oi).dot(Pi)
                Oit = Oi[:Oi.shape[0]-self.channels, :]
                Xi1 = np.linalg.pinv(Oit).dot(Pi1)

                AC = np.vstack((Xi1, Yi)).dot(np.linalg.pinv(Xi))

                A = AC[:order, :order]
                C = AC[order:, :order]

                lamda, psi = np.linalg.eig(A)

                index = np.where(np.imag(lamda) >= 0)
                lamda, psi = lamda[index[0]], psi[:, index[0]]
                index = np.where(np.abs(lamda) >= tol)
                lamda, psi = lamda[index[0]], psi[:, index[0]]

                lamda = np.log(lamda)/self.speriod

                frequencies = np.abs(lamda)/(2*np.pi)
                ratios = -100*np.real(lamda)/np.abs(lamda)
                shapes = C.dot(psi)

                indices = np.argsort(frequencies)
                frequencies = frequencies[indices]
                ratios = ratios[indices]
                shapes = shapes[:, indices]

                return frequencies, ratios, shapes

        self.__getProperties = getProperties
        self.__U = U
        self.__S = S
        self.__V = V



    def __repr__(self):
        method = 'Cov' if self.method == 'cov' else 'Data'
        template = 'Stochastic Subspace Identification (SSI) - {}'
        line = template.format(self.method.capitalize())
        return line


    def plotSingularValues(self):

        """
        Plot the singular values ...
        """

        title = 'Singular values - Stochastic Subspace Indentification (SSI)'
        plt.figure(title, figsize=(10, 3.5))
        plt.title('Singular values', fontsize=12)
        plt.grid()

        plt.plot(np.arange(1, self.__S.shape[0]+1), self.__S, '.', 
                markersize=6, color=(1, 0, 0))
        plt.yscale('log')

        plt.xlabel('Number', fontsize=11)
        plt.ylabel('Value [-]', fontsize=11)

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        plt.show()


    def plotStabilizationDiagram(self, mino=2, maxo=10, ftol=1e-2, dtol=1e-2, 
            mtol=1e-2, minf=0, maxf=1e3, mind=1e-1, maxd=1e1):

        """
        Description

        Parameters
        ----------
        mino: int, optional
            The minimum model order, with default value being 2.
        mixo: int, optional
            The maximum model order, with default value being 10.
        ftol: real, positive, optional
            Relative tolerance for the characterization of stable frequencies.
        dtol: real, positive, optional
            Relative tolerance for the characterization of stable damping 
            ratios.
        mtol: real, positive, optional
            Relative tolerance for the characterization of stable mode shapes.
        minf: real, positive, optional
            The minimum frequency of interest.
        maxf: real, positive, optional
            The maximum frequency of interest.
        mind: real, positive, optional
            The minimum damping ratio of interest.
        maxd: real, positive, optional
            The maximum damping ratio of interest.
        """

        frequencies, dampings, modes = self.getStableModalProperties(mino, 
            maxo, ftol, dtol, mtol, minf, maxf, mind, maxd)

        title = 'Stabilization diagram - Stochastic Subspace Identification'
        plt.figure(title, figsize=(10, 5))
        plt.figure(figsize=(10, 5))
        plt.title('Stabilization diagram', fontsize=12)

        plt.plot(frequencies[0], frequencies[1],
                 'o', markersize=4, fillstyle='none',
                 label='Stable frequency')
        plt.plot(dampings[0], dampings[1],
                 'x', markersize=5, fillstyle='none',
                 label='Stable damping ratio')
        plt.plot(modes[0], modes[1],
                 's', markersize=5, fillstyle='none',
                 label='Stable mode shape')

        plt.xlabel('Frequency [Hz]', fontsize=11)
        plt.ylabel('Model order', fontsize=11)

        plt.xlim([0, maxf])
        plt.ylim([mino, maxo])

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

        return frequencies, dampings, modes
    
    def return_frequencies_dampings_modes(self, mino=2, maxo=10, ftol=1e-2, dtol=1e-2, 
            mtol=1e-2, minf=0, maxf=1e3, mind=1e-1, maxd=1e1):

        """
        Description

        Parameters
        ----------
        mino: int, optional
            The minimum model order, with default value being 2.
        mixo: int, optional
            The maximum model order, with default value being 10.
        ftol: real, positive, optional
            Relative tolerance for the characterization of stable frequencies.
        dtol: real, positive, optional
            Relative tolerance for the characterization of stable damping 
            ratios.
        mtol: real, positive, optional
            Relative tolerance for the characterization of stable mode shapes.
        minf: real, positive, optional
            The minimum frequency of interest.
        maxf: real, positive, optional
            The maximum frequency of interest.
        mind: real, positive, optional
            The minimum damping ratio of interest.
        maxd: real, positive, optional
            The maximum damping ratio of interest.
        """

        frequencies, dampings, modes = self.getStableModalProperties(mino, 
            maxo, ftol, dtol, mtol, minf, maxf, mind, maxd)

        return frequencies, dampings, modes


    def getRealization(self, order):

        """
        Returns
        -------
        system: StateSpace
            The identified stochastic state-space model.
        """

        U1 = self.__U[:, :order]
        S1 = self.__S[:order]
        V1 = self.__V[:order, :]

        if self.method == 'cov':
            sqS1 = np.diag(np.sqrt(S1))
            Oi = U1.dot(sqS1)
            Ci = sqS1.dot(V1)
            Oit = Oi[:Oi.shape[0]-self.channels, :]
            Oib = Oi[self.channels, :]

            A = np.linalg.pinv(Oit).dot(Oib)
            C = Oi[:self.channels, :]
        else:
            pass

        B = np.zeros((A.shape[0], 0))
        D = np.zeros((C.shape[0], 0))

        system = StateSpace(A, B, C, D)

        return system


    def getModalProperties(self, order):

        """
        Description

        Parameters
        ----------
        order: int
            ...

        Returns
        -------
        frequencies: ndarray
            The natrural frequencies.
        ratios: ndarray
            The damping ratios.
        shapes: ndarray
            The mode shapes.
        """

        U1 = self.__U[:, :order]
        S1 = self.__S[:order]
        V1 = self.__V[:order, :]

        frequencies, ratios, shapes = self.__getProperties(U1, S1, V1)

        return frequencies, ratios, shapes




    def getStableModalProperties(self, mino=2, maxo=10, ftol=1e-2, dtol=1e-2, 
            mtol=1e-2, minf=0, maxf=1e3, mind=1e-1, maxd=1e1):

        """
        Description

        Parameters
        ----------
        """

        if mino%2 != 0:
            raise ValueError('Minimum order must be an even number larger \
                             than or equal to 2')

        if maxo%2 != 0 or maxo-mino < 2:
            raise ValueError('Maximum order must be an even number larger \
                             than the Minimum order')

        header = 'Stochastic Subspace Identification (SSI) - {}\n'
        sys.stdout.write(header.format(self.method.capitalize()))
        sys.stdout.write('  calculating stable modal properties...\n')

        properties = []

        for order in range(mino, maxo+2, 2):

            f, d, m = self.getModalProperties(order)
            properties.append([order, f, d, m])


        frequencies = [[], []]
        dampings = [[], [], []]
        modes = [[], [], []]

        # Include a second pass on the properties.

        for p in range(1, len(properties)):
            order = properties[p][0]
            for q in range(len(properties[p][1])):
                freq1 = properties[p][1][q]
                damp1 = properties[p][2][q]
                mode1 = properties[p][3][:, q]

                freq_condition = freq1 < minf or freq1 > maxf
                damp_condition = damp1 < mind or damp1 > maxd

                if freq_condition or damp_condition:
                    continue

                for r in range(len(properties[p-1][1])):
                    freq0 = properties[p-1][1][r]
                    damp0 = properties[p-1][2][r]
                    mode0 = properties[p-1][3][:, r]
                    if np.abs((freq0-freq1)/freq0) <= ftol:
                        frequencies[0].append(freq1)
                        frequencies[1].append(order)
                    if np.abs((damp0-damp1)/damp0) <= dtol:
                        dampings[0].append(freq1)
                        dampings[1].append(order)
                        dampings[2].append(damp1)
                    if (1-MAC(mode0, mode1)) < mtol:
                        modes[0].append(freq1)
                        modes[1].append(order)
                        modes[2].append(mode1)

        sys.stdout.write('  processing completed.\n')

        return frequencies, dampings, modes





class SSICov(object):

    """

    Stochastic Subspace Identification. Computes the state-space model of 
    purely stochastic systems, with no external input, from given output 
    data only.

    Attributes:
    ----------

    Methods:
    ----------

        getSingularValues()
            ...
        getStabilizationDiagram()
            ...
        getRealization(order)
            ...

    """


    def __init__(self, output, speriod, rows, columns=None):

        """
        Parameters:

            output: ndarray
                An array containing the measurements of the ouput vector
                of the stochastic state-space model.
            rows: int
                The number of block rows of the output block Hankel matrix.
            speriod: real
                The sampling period of the output vector.

        """
        
        try:
            output.shape[1]
        except IndexError:
            output = output.reshape((1, output.shape[0]))
            
        self.output = output
        self.rows = rows
        self.sPer = speriod

        samples = int(self.output.shape[1])
        n_out = int(self.output.shape[0])

        if columns == None or columns > samples-rows-1:
            columns = samples-2*rows+1

        denom = np.sqrt(columns)
        
        Y = np.zeros((2*rows*n_out, columns))

        for row in range(2*rows):
            for column, j in enumerate(range(row, row+columns)):
                Y[row*n_out: (row+1)*n_out, column] = self.output[:, j]/denom
        
        Yp = Y[:n_out*rows, :]
        Yf = Y[n_out*rows:, :]
        
        T1i = Yf.dot(Yp.T)

        U, S, V = np.linalg.svd(T1i)
        self.__U = U
        self.__S = S
        self.__V = V



    def getRealization(self, order):
        pass



    def getSingularValues(self):
        
        plt.figure('Singular values - '\
                   'Stochastic Subspace Identification (SSI)',
                    figsize=(10, 3.5))
        plt.title('Singular values', fontsize=12)
        plt.grid()

        plt.plot(np.arange(1, self.__S.shape[0]+1),self.__S,
                 '.', markersize=6, color=(1, 0, 0))
        plt.yscale('log')

        plt.xlabel('Number', fontsize=11)
        plt.ylabel('Value [-]', fontsize=11)

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        plt.show()



    def getPoles(self, minOrder=2, maxOrder=20,
                minFrequency=0, maxFrequency=1e3,
                minDamping=1e-1, maxDamping=1e1):

        if minOrder%2 != 0:
            raise ValueError('Minimum order must be an even number larger \
                             than or equal to 2')

        if maxOrder%2 != 0 or maxOrder-minOrder < 2:
            raise ValueError('Maximum order must be an even number larger \
                             than the Minimum order')

        n_out = int(self.output.shape[0])
        properties = []

        for order in range(minOrder, maxOrder+2, 2):

            U1 = self.__U[:, :order]
            S1 = self.__S[:order]
            V1 = self.__V[:order, :]

            sq_S1 = np.diag(np.sqrt(S1))
            Oi = U1.dot(sq_S1)
            Ci = sq_S1.dot(V1)
            Oit = Oi[:int(Oi.shape[0])-n_out, :]
            Oib = Oi[n_out:, :]
            A = np.linalg.pinv(Oit).dot(Oib)
            C = Oi[:n_out, :]
            
            lamda, psi = np.linalg.eig(A)
            index = np.where(np.imag(lamda) >= 0)
            lamda, psi = lamda[index[0]], psi[:, index[0]]
            lamda = np.log(lamda)/self.sPer

            properties.append([order,
                               np.abs(lamda)/(2*np.pi),
                               -100*np.real(lamda)/np.abs(lamda),
                               C.dot(psi)])

        plt.figure('Natural frequencies - '\
                   'Stochastic Subspace Identification Covariance-driven '\
                   'algorithm (SSI-Cov)', figsize=(10,5))
        plt.title('Natural frequencies', fontsize=12)

        for item in properties:
            plt.plot(item[1], np.repeat(item[0], len(item[1])),
                    'o', markersize=4, fillstyle='none',
                    label='Natural frequencies')

        # plt.plot(frequencies[0], frequencies[1],
        #          'o', markersize=4, fillstyle='none',
        #          label='Stable frequency')
        # plt.plot(dampings[0], dampings[1],
        #          'x', markersize=5, fillstyle='none',
        #          label='Stable damping ratio')
        # plt.plot(modes[0], modes[1],
        #          's', markersize=5, fillstyle='none',
        #          label='Stable mode shape')

        plt.xlabel('Frequency [Hz]', fontsize=11)
        plt.ylabel('Model order', fontsize=11)

        plt.xlim([0, maxFrequency])
        plt.ylim([minOrder, maxOrder])

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        #plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()


    def getStabilizationDiagram(self, minOrder=2, maxOrder=20,
                            frequencyTol=1e-2, dampingTol=5e-2, modeTol=1e-2,
                            minFrequency=0, maxFrequency=1e3,
                            minDamping=1e-1, maxDamping=1e1):

        """
        Stabilization diagram

        Parameters:

            minOrder: int, optional
                Minimum order of the state-space model.
            maxOrder: int, optional
                Maximum order of the state-space model.
            frequencyTol: real, optional
                Tolerance of the stability criterion for the frequencies.
            dampingTol: real, optional
                Tolerance of the stability criterion for the damping 
                ratios.
            modeTol: real, optional
                Tolerance of the stability criterion for the mode shapes.
            minFrequency: real, optional
                Minimum frequency
            maxFrequency: real, optional
                Maximum frequency
            minDamping: real, optional
                Minimum damping
            maxdamping: real, optional
                Maximum damping

        Returns:

            fs: list
                List of two lists containing the stable frequencies and
                their model order respectively.
            ks: list
                List of three lists containing the frequencies that 
                correspond to stable damping ratios, the stable damping
                ratios and their model order respectively.
            ms: list
                List of three lists containing the frequencies that 
                correspond to stable mode shapes, the stable mode shape
                and their model order respectively.

        """

        if minOrder%2 != 0:
            raise ValueError('Minimum order must be an even number larger \
                             than or equal to 2')

        if maxOrder%2 != 0 or maxOrder-minOrder < 2:
            raise ValueError('Maximum order must be an even number larger \
                             than the Minimum order')

        n_out = int(self.output.shape[0])
        properties = []

        for order in range(minOrder, maxOrder+2, 2):

            U1 = self.__U[:, :order]
            S1 = self.__S[:order]
            V1 = self.__V[:order, :]

            sq_S1 = np.diag(np.sqrt(S1))
            Oi = U1.dot(sq_S1)
            Ci = sq_S1.dot(V1)
            Oit = Oi[:int(Oi.shape[0])-n_out, :]
            Oib = Oi[n_out:, :]
            A = np.linalg.pinv(Oit).dot(Oib)
            C = Oi[:n_out, :]
            
            lamda, psi = np.linalg.eig(A)
            index = np.where(np.imag(lamda) >= 0)
            lamda, psi = lamda[index[0]], psi[:, index[0]]
            lamda = np.log(lamda)/self.sPer

            properties.append([order,
                               np.abs(lamda)/(2*np.pi),
                               -100*np.real(lamda)/np.abs(lamda),
                               C.dot(psi)])
        
        frequencies = [[], []]
        dampings = [[], [], []]
        modes = [[], [], []]

        for p in range(1, len(properties)):
            order = properties[p][0]
            for q in range(len(properties[p][1])):
                freq1 = properties[p][1][q]
                damp1 = properties[p][2][q]
                mode1 = properties[p][3][:, q]
                
                freq_condition = freq1 < minFrequency or freq1 > maxFrequency
                damp_condition = damp1 < minDamping or damp1 > maxDamping

                if freq_condition or damp_condition:
                    continue

                for r in range(len(properties[p-1][1])):
                    freq0 = properties[p-1][1][r]
                    damp0 = properties[p-1][2][r]
                    mode0 = properties[p-1][3][:, r]
                    if np.abs((freq0-freq1)/freq0) <= frequencyTol:
                        frequencies[0].append(freq1)
                        frequencies[1].append(order)
                    if np.abs((damp0-damp1)/damp0) <= dampingTol:
                        dampings[0].append(freq1)
                        dampings[1].append(order)
                        dampings[2].append(damp1)
                    if (1-MAC(mode0, mode1)) < modeTol:
                        modes[0].append(freq1)
                        modes[1].append(order)
                        modes[2].append(mode1)

        plt.figure('Stabilization diagram - '\
                   'Stochastic Subspace Identification Covariance-driven '\
                   'algorithm (SSI-Cov)', figsize=(10,5))
        plt.title('Stabilization diagram', fontsize=12)

        plt.plot(frequencies[0], frequencies[1],
                 'o', markersize=4, fillstyle='none',
                 label='Stable frequency')
        plt.plot(dampings[0], dampings[1],
                 'x', markersize=5, fillstyle='none',
                 label='Stable damping ratio')
        plt.plot(modes[0], modes[1],
                 's', markersize=5, fillstyle='none',
                 label='Stable mode shape')

        plt.xlabel('Frequency [Hz]', fontsize=11)
        plt.ylabel('Model order', fontsize=11)

        plt.xlim([0, maxFrequency])
        plt.ylim([minOrder, maxOrder])

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

        return properties




class SSIData(object):

    def __init__(self, output, rows, speriod, columns=None, algorithm='UPC'):

        """
        Data-driven algorithm for Stochastic Subspace Identification

        Parameters:

            output: ndarray
                An array containing the measurements of the ouput vector
                of the stochastic state-space model.
            rows: int
                The number of block rows of the output block Hankel matrix.
            speriod: real
                The sampling period of the output vector.
            columns: int, optional
                The number of columns of the output block Hankel matrix
            algorithm: {'PC', 'UPC', 'CVA'}


        """

        try:
            output.shape[1]
        except IndexError:
            output = output.reshape((1, output.shape[0]))

        if algorithm not in ['PC', 'UPC', 'CVA']:
           raise ValueError('Invalid ...')

        self.output = output
        self.samp_per = speriod
        self.rows = rows


        samples = int(self.output.shape[1])
        n_out = int(self.output.shape[0]) 
        columns = samples-2*rows+1
        den = np.sqrt(columns)

        Y = np.zeros((2*rows*n_out, columns))

        for row in range(2*rows):
            for column, i in enumerate(range(row, row+columns)):
                Y[row*n_out: (row+1)*n_out, column] = self.output[:, i]/den

        Yp = Y[:rows*n_out, :]
        Yf = Y[rows*n_out:, :]


        Q, R = np.linalg.qr(Y.T)
        Q, R = Q.T, R. T
        self.__Q = Q
        self.__R = R

        if algorithm == 'PC':
            # Use of QR decomposition for W2
            R11 = R[:rows*n_out, :rows*n_out]
            Cov_YpYp = R11.dot(R11.T)
            W1 = np.eye(rows*n_out)
            W2 = Yp.T.dot(np.linalg.inv(sp.linalg.sqrtm(Cov_YpYp))).dot(Yp)
        elif algorithm == 'UPC':
            W1 = np.eye(rows*n_out)
            W2 = np.eye(columns)
        elif algorithm == 'CVA':
            # Use of QR decomposition for W1
            W1 = np.linalg.inv(sp.linalg.sqrtm(Yf.dot(Yf.T)))
            W2 = np.eye(columns)


        Ob_proj = R[rows*n_out:, :rows*n_out].dot(Q[:rows*n_out, :])
        Ob_proj = W1.dot(Ob_proj).dot(W2)

        U, S, V = np.linalg.svd(Ob_proj)
        self.__U = U
        self.__S = S
        self.__V = V


    def getSingularValues(self):
        
        plt.figure('Stochastic Subspace Identification Data-driven '\
                   'algorithm (SSI-Data) - '\
                   'Singular values', figsize=(10, 3.5))
        plt.title('Singular values', fontsize=12)
        plt.grid()

        plt.plot(np.arange(1, self.__S.shape[0]+1),self.__S,
                 '.', markersize=6, color=(1, 0, 0))
        plt.yscale('log')

        plt.xlabel('Number', fontsize=11)
        plt.ylabel('Value [-]', fontsize=11)

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        plt.show()


    def getStabilizationDiagram(self, minOrder=2, maxOrder=20,
                            freqTol=1e-2, dampTol=5e-2, modeTol=1e-2,
                            minFrequency=0, maxFrequency=1e3,
                            minDamping=1e-1, maxDamping=1e1):
        
        """
        Parameters:

            minOrder: int, optional
                Minimum order 
            maxOrder: int, optional
                Maximum order
            freqTol: real, optional
                Relative tolerance for the characterization of stable 
                frequencies.
            dampTol: real, optional
                Relative tolerance for the characterization of stable
                damping ratios.
            modeTol: real, optional
                Relative tolerance for the characterization of stable
                mode shapes.
            minFrequency: real, optional
                ...
            maxFrequency: real, optional
                ...
            minDamping: real, optional
                ...
            maxDamping: real, optional
                ...

        Returns:



        """

        if minOrder%2 != 0:
            raise ValueError('Minimum order must be an even number larger \
                             than or equal to 2')

        if maxOrder%2 != 0 or maxOrder-minOrder < 2:
            raise ValueError('Maximum order must be an even number larger \
                             than the minimum order')

        n_out = int(self.output.shape[0])
        properties = []

        rows = self.rows
        R = self.__R
        Q = self.__Q

        Pi_1 = R[(rows+1)*n_out:, :(rows+1)*n_out].dot(
               Q[:(rows+1)*n_out, :])
        Yi = R[rows*n_out:(rows+1)*n_out, :(rows+1)*n_out].dot(
             Q[:(rows+1)*n_out])

        for order in range(minOrder, maxOrder+2, 2):

            U1 = self.__U[:, :order]
            S1 = self.__S[:order]
            V1 = self.__V[:order, :]

            sq_S1 = np.diag(np.sqrt(S1))

            Pi = U1.dot(np.diag(S1)).dot(V1)
            Oi = U1.dot(sq_S1)
            Xi = np.linalg.pinv(Oi).dot(Pi)
            Oit = Oi[:int(Oi.shape[0])-n_out, :]
            Xi_1 = np.linalg.pinv(Oit).dot(Pi_1)

            AC = np.vstack((Xi_1, Yi)).dot(np.linalg.pinv(Xi))

            A = AC[:order, :order]
            C = AC[order:, :order]

            lamda, psi = np.linalg.eig(A)
            index = np.where(np.imag(lamda) >= 0)
            lamda, psi = lamda[index[0]], psi[:, index[0]]
            lamda = np.log(lamda)/ self.samp_per

            properties.append([order,
                               np.abs(lamda)/(2*np.pi),
                               -100*np.real(lamda)/np.abs(lamda),
                               C.dot(psi)])

        frequencies = [[], []]
        dampings = [[], [], []]
        modes = [[], [], []]

        for p in range(1, len(properties)):
            order = properties[p][0]
            for q in range(len(properties[p][1])):
                freq1 = properties[p][1][q]
                damp1 = properties[p][2][q]
                mode1 = properties[p][3][:, q]
                
                freq_condition = freq1 < minFrequency or freq1 > maxFrequency
                damp_condition = damp1 < minDamping or damp1 > maxDamping

                if freq_condition or damp_condition:
                    continue

                for r in range(len(properties[p-1][1])):
                    freq0 = properties[p-1][1][r]
                    damp0 = properties[p-1][2][r]
                    mode0 = properties[p-1][3][:, r]
                    if np.abs((freq0-freq1)/freq0) <= freqTol:
                        frequencies[0].append(freq1)
                        frequencies[1].append(order)
                    if np.abs((damp0-damp1)/damp0) <= dampTol:
                        dampings[0].append(freq1)
                        dampings[1].append(order)
                        dampings[2].append(damp1)
                    if (1-MAC(mode0, mode1)) < modeTol:
                        modes[0].append(freq1)
                        modes[1].append(order)
                        modes[2].append(mode1)

        plt.figure('Stabilization diagram - '\
                   'Stochastic Subspace Identification Data-driven '\
                   'algorithm (SSI-Data)', figsize=(10,5))
        plt.title('Stabilization diagram', fontsize=12)

        plt.plot(frequencies[0], frequencies[1],
                 'o', markersize=4, fillstyle='none',
                 label='Stable frequency')
        plt.plot(dampings[0], dampings[1],
                 'x', markersize=5, fillstyle='none',
                 label='Stable damping ratio')
        plt.plot(modes[0], modes[1],
                 's', markersize=5, fillstyle='none',
                 label='Stable mode shape')

        plt.xlabel('Frequency [Hz]', fontsize=11)
        plt.ylabel('Model order', fontsize=11)

        plt.xlim([0, maxFrequency])
        plt.ylim([minOrder, maxOrder])

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

        return frequencies, dampings, modes







class ERA:

    """
    Eigensystem Realization Algorithm (ERA) for modal parameter identification 
    of dynamic systems from test data [1].

    Parameters
    ----------
    cols: int, optional
        The number of columns of the output block Hankel matrix. If not 
        specified, all given data samples are used for the construction 
        of the block Hankel matrices.
    inpt: ndarray, optional
        An array containing the data samples of the measured inputs.
    output: ndarray
        An array containing the data samples of the measured outputs.
    rows: int
        The number of block rows of the output block Hankel matrix.
    speriod: real, positive
        The sampling period of the output and input vector.

    Attributes
    ----------

    Methods:
    ----------
    
    References
    ----------
     .. [1] J. Y. Juang, R. S. Pappa, "An Eigensystem Realization Algorithm
        for Modal Parameter Identification and Model Reduction", Journal of
        Guidance, Control and Dynamics, Vol. 8, pp. 620-627, 1985.

     .. [2] J. M. Caicedo, "Practical guidelines for the natural excitation
        technique (NExT) and the eigensystem realization algorithm (ERA) for
        modal identification using ambient vibration", Experimental 
        Techniques, Vol. 34, pp. 52-58, 2010.

    Examples
    --------
    >>> 
    """

    def __init__(self, output, speriod, rows, columns=None, inpt=None):
        
        try:
            output.shape[1]
        except IndexError:
            output = output.reshape((1, output.shape[0]))

        # normalize output with respect to the maximum input value
        # assuming that the system is linear

        self.input = inpt
        self.output = output
        self.speriod = speriod

        samples = int(self.output.shape[1])
        n_out = int(self.output.shape[0])

        if columns == None or columns > samples-rows-1:
            columns = samples-rows-1 # !!!!!!!
        
        H1 = np.zeros((rows*n_out, columns))
        H2 = np.zeros((rows*n_out, columns))

        for row in range(rows): 
            for column, li in enumerate(range(row, row+columns)):
                H1[row*n_out: (row+1)*n_out, column] = self.output[:, li]
                H2[row*n_out: (row+1)*n_out, column] = self.output[:, li+1]

        self.__H1 = H1
        self.__H2 = H2

        U, S, V = np.linalg.svd(H1)

        self.__U = U
        self.__S = S
        self.__V = V


    def getFlexibilityMatrix(self, order, eigs):

        pass


    def getTransferFunction(self, order, omega, eigs):

        """

        Parameters:

            order: int, optional
                ...

        Returns:

            tf:
                ...

        """

        A, B, C = self.getRealization(order, domain='continuous')

        tF = np.zeros(C.shape[0], dtype=complex)
        evals, evecs = np.linalg.eig(A)
        inv_evecs = np.linalg.inv(evecs)

        for index in range(len(evals)):
            tF -= C.dot(evecs[:, index].dot(inv_evecs[index, :])).dot(B)

        return tF



    def getRealization(self, order, domain='discrete'):

        """
        Description

        Parameters:
        ----------
        order: int, optional
            An even number representing the order of the state-space model.
        domain: {'continuous', 'discrete'}, optional
            The time-domain in which the realization of the state-space
            model is performed.

        Returns:
        ----------
        A: ndarray, shape (order, order)
            The system matrix.
        B: ndarray, shape (order, 1)
            The input matrix.
        C: ndarray, shape (num_out, order)
            The output influence matrix.

        Raises
        ------
        """

        U1 = self.__U[:, :order]
        S1 = self.__S[:order]
        V1 = self.__V[:order, :]

        inv_sq_S1 = np.diag(1/np.sqrt(S1))
        num_out = int(self.output.shape[0])

        A = inv_sq_S1.dot(U1.T).dot(self.__H2).dot(V1.T).dot(inv_sq_S1)
        B = np.diag(S1).dot(V1)[:, 1]
        C = U1.dot(inv_sq_S1)[:num_out, :]

        if domain == 'continuous':
            evals, evecs = np.linalg.eig(A)
            log_evals = np.diag(np.log(evals))
            inv_evecs = np.linalg.inv(evecs)

            temp = evecs.dot(log_evals).dot(inv_evecs)/self.speriod

            B = temp.dot(np.linalg.inv(A-np.eye(order))).dot(B)
            A = temp
        elif domain == 'discrete':
            pass
        else:
            raise ValueError('Domain should be either continuous or discrete')

        return A, B, C

    def getSystemProperties(self, order, domain='discrete'):
        pass


    def simulation(self):
        pass


    def getStabilizationDiagram(self, mino=2, maxo=20, ftol=1e-2, dtol=5e-2, 
                mtol=1e-2, minf=0, maxf=1e3, mind=1e-1, maxd=1e1):

        """
        
        Parameters:
        ----------
        mino: int, optional
            Minimum order .
        maxo: int, optional
            Maximum order.
        ftol: real, optional
            Relative tolerance for the characterization of stable 
            frequencies.
        dtol: real, optional
            Relative tolerance for the characterization of stable
            damping ratios.
        mtol: real, optional
            Relative tolerance for the characterization of stable
            mode shapes.
        minf: real, optional
            ...
        maxf: real, optional
            ...
        mind: real, optional
            ...
        maxd: real, optional
            ...

        Returns:



        """

        if mino%2 != 0:
            raise ValueError('Minimum order must be an even number larger \
                             than or equal to 2')

        if maxo%2 != 0 or maxo-mino < 2:
            raise ValueError('Maximum order must be an even number larger \
                             than the Minimum order')


        num_out = int(self.output.shape[0])
        properties = []

        for order in range(mino, maxo+2, 2):

            U1 = self.__U[:, :order]
            S1 = self.__S[:order]
            V1 = self.__V[:order, :]

            inv_sq_S1 = np.diag(1/np.sqrt(S1))

            A = inv_sq_S1.dot(U1.T).dot(self.__H2).dot(V1.T).dot(inv_sq_S1)
            B = np.diag(S1).dot(V1)[:, :num_out]
            C = U1.dot(inv_sq_S1)[:num_out, :]#U1[:nOut, :]

            lamda, psi = np.linalg.eig(A)
            index = np.where(np.imag(lamda)>=0)
            lamda, psi = lamda[index[0]], psi[:, index[0]]
            lamda = np.log(lamda)/self.speriod

            properties.append([order,
                               np.abs(lamda)/(2*np.pi),
                               -100*np.real(lamda)/np.abs(lamda),
                               C.dot(psi)])

        frequencies = [[], []]
        dampings = [[], [], []]
        modes = [[], [], []]

        for p in range(1, len(properties)):
            order = properties[p][0]
            for q in range(len(properties[p][1])):
                freq1 = properties[p][1][q]
                damp1 = properties[p][2][q]
                mode1 = properties[p][3][:, q]

                freq_condition = freq1 < minf or freq1 > maxf
                damp_condition = damp1 < mind or damp1 > maxd

                if freq_condition or damp_condition:
                    continue

                for r in range(len(properties[p-1][1])):
                    freq0 = properties[p-1][1][r]
                    damp0 = properties[p-1][2][r]
                    mode0 = properties[p-1][3][:, r]
                    if np.abs((freq0-freq1)/freq0) <= ftol:
                        frequencies[0].append(freq1)
                        frequencies[1].append(order)
                    if np.abs((damp0-damp1)/damp0) <= dtol:
                        dampings[0].append(freq1)
                        dampings[1].append(order)
                        dampings[2].append(damp1)
                    if (1-MAC(mode0, mode1)) <= mtol:
                        modes[0].append(freq1)
                        modes[1].append(order)
                        modes[2].append(mode1)


        plt.figure('Eigensystem Realization Algorithm (ERA) - '\
                   'Stabilization diagram', figsize=(10,5))
        plt.title('Stabilization diagram', fontsize=12)

        plt.plot(frequencies[0], frequencies[1],
                 'o', markersize=4, fillstyle='none',
                 label='Stable frequency')

        plt.plot(dampings[0], dampings[1],
                 'x', markersize=5, fillstyle='none',
                 label='Stable damping ratio')

        plt.plot(modes[0], modes[1],
                 's', markersize=5, fillstyle='none',
                 label='Stable mode shape')

        plt.xlabel('Frequency [Hz]', fontsize=11)
        plt.ylabel('Model order', fontsize=11)

        plt.xlim([0, maxFrequency])
        plt.ylim([mino, maxo])

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

        # plt.close()

        #return frequencies, dampings, modes


    # def principalAngles(self):
        
    #     # check first if CVA algorithm is used !!!

    #     plt.figure('Principal angles', figsize=(8, 2.5))

    #     plt.title('Principal angles')
    #     plt.xlabel('Model order')
    #     plt.ylabel('Principal angle [deg]')

    #     plt.plot(np.arccos(self.__S)*180/np.pi, '.', color=(0, 0, 1))

    #     plt.tight_layout()
    #     plt.show()


    def getSingularValues(self):
        
        plt.figure('Eigensystem Realization Algorithm (ERA) - '\
                   'Singular values',
                    figsize=(10, 3.5))
        plt.title('Singular values', fontsize=12)
        plt.grid()

        plt.plot(np.arange(1, self.__S.shape[0]+1),self.__S,
                 '.', markersize=6, color=(1, 0, 0))
        plt.yscale('log')

        plt.xlabel('Number', fontsize=11)
        plt.ylabel('Value [-]', fontsize=11)

        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        plt.show()
