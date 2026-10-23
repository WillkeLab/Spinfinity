"""Helper functions for processing simulated data."""

import numpy as np
from scipy import fft
from scipy import signal as scipy_signal
from scipy.optimize import curve_fit


def calc_fft(
    times,
    signal,
    *,
    one_sided=True,
    window=None,
    detrend="constant",
    normalization="amplitude",
):
    """Compute a Fourier spectrum with optional windowing and amplitude scaling.

    Parameters
    ----------
    times : array_like
        Finite, uniformly spaced, strictly increasing times, shape (N,).
        At least two samples are required. Times in ns give frequencies in GHz.
    signal : array_like
        Finite signal values, shape (N,). Complex values require one_sided=False.
    one_sided : bool, optional
        Return non-negative frequencies for a real signal if True (default).
        False returns the full spectrum in FFT order, including negative frequencies.
    window : {None, "hann"}, optional
        None uses a rectangular window (default). "hann" uses a periodic Hann window.
    detrend : {None, "constant", "linear"}, optional
        Remove nothing, the mean (default), or a least-squares straight line.
        Detrending is applied before windowing.
    normalization : {"amplitude", "raw"}, optional
        "amplitude" (default) divides by the window sum and doubles one-sided
        interior bins. "raw" returns unscaled FFT coefficients.

    Returns
    -------
    frequencies : numpy.ndarray
        Frequencies in inverse time units, shape (N // 2 + 1,) for a one-sided
        spectrum or (N,) otherwise. Times in ns give GHz, not rad/ns.
    spectrum : numpy.ndarray
        Complex coefficients with the same shape as frequencies.
        Use np.abs(spectrum) for magnitudes. Amplitude scaling retains the
        signal units; raw magnitudes also depend on sample count and window.

    Raises
    ------
    ValueError
        If inputs are not matching finite one-dimensional arrays with at least
        two samples, times are not increasing and uniform, a complex signal
        is used with one_sided=True, or an option is unsupported.

    Notes
    -----
    With amplitude scaling, an isolated bin-centered real cosine of amplitude A,
    away from DC and Nyquist, gives magnitude A in a one-sided spectrum or A/2
    at each signed frequency in a full spectrum. DC and the even-N Nyquist bin
    are not doubled.
    Window correction does not recover the initial amplitude of a decaying
    oscillation or remove off-bin leakage. This is not a power spectral density.

    Phases are relative to the first sample. Inputs are not modified.
    No peak finding, background fitting, or normalization to the maximum is done.
    """
    times = np.asarray(times, dtype=float)
    signal = np.asarray(signal)

    if (
        times.ndim != 1
        or signal.ndim != 1
        or times.size != signal.size
        or times.size < 2
    ):
        raise ValueError("times and signal must be matching 1D arrays with N >= 2.")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(signal)):
        raise ValueError("times and signal must contain only finite values.")

    dt = times[1] - times[0]
    if dt <= 0 or not np.allclose(np.diff(times), dt, rtol=1e-7, atol=0):
        raise ValueError("times must be strictly increasing and uniformly spaced.")
    if one_sided and np.iscomplexobj(signal):
        raise ValueError("Complex signals require one_sided=False.")
    if (
        window not in (None, "hann")
        or detrend not in (None, "constant", "linear")
        or normalization not in ("amplitude", "raw")
    ):
        raise ValueError("Unsupported window, detrend, or normalization.")

    n = times.size
    weights = np.ones(n) if window is None else scipy_signal.windows.hann(n, sym=False)
    processed = (
        signal if detrend is None else scipy_signal.detrend(signal, type=detrend)
    )
    weighted = processed * weights

    if one_sided:
        spectrum = fft.rfft(weighted)
        frequencies = fft.rfftfreq(n, d=dt)
    else:
        spectrum = fft.fft(weighted)
        frequencies = fft.fftfreq(n, d=dt)

    if normalization == "amplitude":
        spectrum /= weights.sum()
        if one_sided:
            spectrum[1:] *= 2
            if n % 2 == 0:
                spectrum[-1] /= 2

    return frequencies, spectrum


# region Fitting functions


def fit_fano(x, y, outputN: int = None):
    """
    Fit a symmetric Fano line with a constant offset.

    Function: f(x) = C + A / (1 + 4 * ((x - x0) / FWHM)**2)

    Parameters
    ----------
    x : array_like
        Finite, strictly increasing values of the independent variable.
        At least five samples are required.
    y : array_like
        Finite, nonconstant values corresponding to ``x``. Peaks and dips
        are supported.
    outputN : int, optional
        Number of evenly spaced points in the fitted output. Defaults to the
        number of input samples.

    Returns
    -------
    xfit : numpy.ndarray
        ``outputN`` evenly spaced points covering the range of ``x``.
    yfit : numpy.ndarray
        Fitted model evaluated at ``xfit``, in the units of ``y``.
    popt : numpy.ndarray
        Optimal parameters ``[A, x0, FWHM, C]``. The signed amplitude
        ``A`` is the peak height relative to ``C``. The center ``x0``
        and positive full width at half maximum ``FWHM`` have the units
        of ``x``; ``A`` and ``C`` have the units of ``y``.
    perr : numpy.ndarray
        One-standard-deviation uncertainties
        ``[A_err, x0_err, FWHM_err, C_err]``, in the corresponding units.

    Raises
    ------
    ValueError
        If the inputs are not matching one-dimensional arrays with at least
        five samples, contain nonfinite values, have nonincreasing ``x``,
        or have constant ``y``.

    Notes
    -----
    This symmetric convention is a Lorentzian; no asymmetry parameter is
    fitted. The center is restricted to the sampled range.
    Data are scaled internally to handle small currents, but returned
    amplitudes and uncertainties retain their original units.
    Convergence errors from ``scipy.optimize.curve_fit`` are propagated.
    """

    def fano_model(t, A, x0, FWHM, C):
        return C + A / (1 + 4 * ((t - x0) / FWHM) ** 2)

    if outputN is None:
        outputN = len(x)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 5:
        raise ValueError(
            "x and y must be matching 1D arrays with at least five samples."
        )
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must contain only finite values.")
    if np.any(np.diff(x) <= 0) or np.ptp(y) == 0:
        raise ValueError("x must be strictly increasing and y must be nonconstant.")

    # Scale small signals before fitting.
    scale = np.max(np.abs(y))
    y_scaled = y / scale

    # Initial guesses, including the sign of a peak or dip.
    C0 = (y_scaled[0] + y_scaled[-1]) / 2
    peak_index = np.argmax(np.abs(y_scaled - C0))
    A0 = y_scaled[peak_index] - C0
    x0 = x[peak_index]
    FWHM0 = (x[-1] - x[0]) / 10

    p0 = [A0, x0, FWHM0, C0]
    bounds = ([-np.inf, x[0], 0, -np.inf], [np.inf, x[-1], np.inf, np.inf])

    popt, pcov = curve_fit(fano_model, x, y_scaled, p0=p0, bounds=bounds, maxfev=20000)

    parameter_scale = np.array([scale, 1.0, 1.0, scale])
    popt = popt * parameter_scale
    perr = np.sqrt(np.diag(pcov)) * parameter_scale

    xfit = np.linspace(x[0], x[-1], outputN)
    yfit = fano_model(xfit, *popt)

    return xfit, yfit, popt, perr


def fit_exponential(x, y, outputN: int = None):
    """
    Fit an exponential decay with a constant offset.

    Function: f(x) = C + A * exp(-x / T)

    Parameters
    ----------
    x : array_like
        Values of the independent variable.
    y : array_like
        Measured values corresponding to ``x``.
    outputN : int, optional
        Number of evenly spaced points in the fitted output. Defaults to the
        number of input samples.

    Returns
    -------
    xfit : numpy.ndarray
        ``outputN`` evenly spaced points covering the range of ``x``.
    yfit : numpy.ndarray
        Fitted model evaluated at ``xfit``.
    popt : numpy.ndarray
        Optimal parameters ``[A, T, C]``.
    perr : numpy.ndarray
        One-standard-deviation uncertainties ``[A_err, T_err, C_err]``.
    """

    def exp_model(t, A, T, C):
        return C + A * np.exp(-t / T)

    if outputN is None:
        outputN = len(x)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Initial guesses
    A0 = y[0] - y[-1]
    T0 = (x[-1] - x[0]) / 2
    C0 = y[-1]

    p0 = [A0, T0, C0]

    popt, pcov = curve_fit(exp_model, x, y, p0=p0, maxfev=20000)

    perr = np.sqrt(np.diag(pcov))

    xfit = np.linspace(np.min(x), np.max(x), outputN)
    yfit = exp_model(xfit, *popt)

    return xfit, yfit, popt, perr


def fit_oscillation(x, y, outputN: int = None):
    """
    Fit an exponentially damped cosine with a constant offset.

    Function: f(x) = C + A * exp(-x / T) * cos(2 * pi * f * x + phi)

    Parameters
    ----------
    x : array_like
        Increasing, evenly spaced values of the independent variable. At least
        five samples are required.
    y : array_like
        Measured values corresponding to ``x``.
    outputN : int, optional
        Number of evenly spaced points in the fitted output. Defaults to the
        number of input samples.

    Returns
    -------
    xfit : numpy.ndarray
        ``outputN`` evenly spaced points covering the range of ``x``.
    yfit : numpy.ndarray
        Fitted model evaluated at ``xfit``.
    popt : numpy.ndarray
        Optimal parameters ``[A, T, f, phi, C]``.
    perr : numpy.ndarray
        One-standard-deviation uncertainties
        ``[A_err, T_err, f_err, phi_err, C_err]``.

    Notes
    -----
    The initial frequency estimate is the strongest nonzero component of the
    real-valued discrete Fourier transform of ``y``.
    """

    def oscillation_model(t, A, T, f, phi, C):
        return C + A * np.exp(-t / T) * np.cos(2 * np.pi * f * t + phi)

    if outputN is None:
        outputN = len(x)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Initial guesses
    A0 = (np.max(y) - np.min(y)) / 2
    T0 = (x[-1] - x[0]) / 2
    C0 = np.mean(y)

    dt = np.mean(np.diff(x))
    frequencies = np.fft.rfftfreq(len(x), d=dt)
    spectrum = np.abs(np.fft.rfft(y - C0))
    f0 = frequencies[np.argmax(spectrum[1:]) + 1]
    phi0 = 0

    p0 = [A0, T0, f0, phi0, C0]

    popt, pcov = curve_fit(oscillation_model, x, y, p0=p0, maxfev=20000)

    perr = np.sqrt(np.diag(pcov))

    xfit = np.linspace(np.min(x), np.max(x), outputN)
    yfit = oscillation_model(xfit, *popt)

    return xfit, yfit, popt, perr


# endregion
