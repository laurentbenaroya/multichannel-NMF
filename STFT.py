import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack


def Stft(x, wlen, h, nfft, fs, winType='hamming'):
    '''
    Compute the Short-Time Fourier Transform of a signal, with an hamming window.

    Parameters
    ----------
    x : numpy array (n x 1)
        vector signal for which we want to compute the STFT
    wlen : int
        length of the window, in samples
    h : int
        hop size, in samples
    nfft : int
        number of samples used to compute the FFT algorithm
    fs : int
        sampling frequency
    winType : char*
         type of window used

    Returns
    -------
    X : numpy array (F x N)
        STFT of the input signal
    f : numpy array (F,)
        frequency vector (Hz)
    t : numpy array (N,)
        time vector of the center of the frames (seconds)

    AUTHORS
    ======
    bouvier@ircam.fr, based on the 'stft.m' function from Hristo Zhivomirov (dated 12/21/13).
    Copyright (c) 2015 IRCAM under GNU General Public License version 3.
    Developed on MatLab 7.7.0.471 (R2008b)

    Translation to python, E. L. Benaroya - 09/2018 - Telecom ParisTech - laurent.benaroya@gmail.com
    '''

    if winType.lower() == 'hamming':
        win = np.hamming(wlen)
    else:
        raise Exception("window type Not implemented yet")

    xlen = len(x)
    rown = int(np.ceil((1+nfft)/2))
    coln = 1+int(np.floor((xlen-wlen)/h))
    X = np.zeros((rown, coln), dtype=np.complex128)

    indx = 0
    col = 0

    while indx+wlen < xlen:
        xw = x[indx:(indx+wlen)]*win
        Xtmp = scipy.fftpack.fft(xw, nfft)

        X[:, col] = Xtmp[:rown]

        indx += h
        col += 1

    t = np.arange(int(wlen/2), xlen-int(wlen/2), h)/fs
    f = np.arange(rown)*fs/nfft

    return X, f, t


def Istft(X, wlen, h, nfft, fs, winType='hamming'):
    '''
    Inverse STFT
    Parameters
    ----------
    X : numpy array (F, N)
    wlen : int
        window length in samples
    h : int
        offset in samples
    nfft : int
        number of samples used to compute the IFFT algorithm
    fs : int
        sampling frequency
    winType : char*
        analysis and synthesis overlap add window type

    Returns
    -------
    x : numpy array (n,)
        output waveform
    t : numpy array (n,)
        time vector in seconds
    '''

    if winType.lower() == 'hamming':
        win = np.hamming(wlen)
    else:
        raise Exception("window type Not implemented yet")

    rown, coln = X.shape
    xlen = wlen + (coln-1) * h

    x = np.zeros((xlen,))
    tmp = np.zeros((xlen,))

    if int(nfft/2)*2 != nfft:  # nfft is odd, exclude Nyquist frequency
        for b in np.arange(0, h*coln, h):
            Xtmp = X[:, int(b/h)]

            xprim = np.concatenate((Xtmp, np.flip(np.conj(Xtmp[1:]), axis=0)), axis=0)
            xprim = np.real(scipy.fftpack.ifft(xprim, nfft))
            xprim = xprim[:wlen]

            x[b:(b+wlen)] += xprim*win
            tmp[b:(b+wlen)] += win*win

    else:
        for b in np.arange(0, h*coln, h):
            Xtmp = X[:, int(b/h)]

            xprim = np.concatenate((Xtmp, np.flip(np.conj(Xtmp[1:-1]), axis=0)), axis=0)
            xprim = np.real(scipy.fftpack.ifft(xprim, nfft))
            xprim = xprim[:wlen]

            x[b:(b+wlen)] += xprim*win
            tmp[b:(b+wlen)] += win*win

    x = x/tmp

    actlen = len(x)
    t = np.arange(0, actlen)/fs
    return x, t


def spectrogram(x,  wlen, h, nfft, fs, dB=True, plot=False, show=False):
    '''
    Compute and display the spectrogram of a 1D signal
    Parameters
    ----------
    x       : np array (n,)
        signal for which we want to compute (and plot) the spectrogram
    wlen    : int
        length of the analysis window, in samples
    h       : int
        hop size, in samples
    nfft    : int
        number of samples used to compute the FFT
    fs      : int
        sampling frequency
    dB : bool
        take 20*log10(spectrogram) if True
    plot : bool
        plot spectrogram if True
    show : bool
        call plt.show() if True

    Returns
    -------
    Xspec : numpy array (F x N)
        spectrogram
    f : np array (F,)
        frequency vector (Hz)
    t : np array (N,)
        time vector of the center of the frames (seconds)
    E. L. Benaroya - 09/2018 - Telecom ParisTech - laurent.benaroya@gmail.com
    '''

    X, f, t = Stft(x, wlen, h, nfft, fs)
    Xspec = np.abs(X)
    if dB:
        Xspec = 20*np.log10(Xspec)  # in dB
    if plot:
        plt.imshow(Xspec, origin='lower', interpolation='nearest', aspect='auto', extent=(0, t[-1], 0, f[-1]))
        # plt.colorbar()
        plt.xlabel('Temps (secondes)')
        plt.ylabel('Fréquence (Hz)')
    if show:
        plt.show()

    return Xspec, f, t
