"""Iterative prewhitening: split a light curve into coherent frequency components.

The CNN sees one phase fold at a time and was trained on stars whose light
curve IS that single variation. On a multiperiodic star the fold at any one
period carries every other variation as scatter, so the image the network gets
is not the kind of image its training set contains. Removing the other
components before folding puts the star back inside that distribution — which
is the only way to ask the network about one variation at a time.

Everything runs on relative flux in ppt ((f/mean - 1) * 1e3), the same scaling
`periodograms.ls_periodogram` uses internally, so amplitudes are comparable
between stars and the least-squares fit is not dominated by the mean level.

Prewhitening a BROAD feature does not return its true content. A stochastic or
quasi-periodic hump wider than the resolution gets shredded into a comb of
peaks spaced by about one Rayleigh width, because each subtraction leaves a
residual the next iteration finds one resolution element away. Components
spaced by ~1/T inside a single cluster mark where the hump is; they are not
evidence of that many independent modes. `cluster_components` is what flags
the situation.

The frequency grid starts at 1/baseline instead of the pipeline's 2/baseline:
a single long-term wave over the sector is a real component here (it is what
Labadie-Bartz+ 2022 flag as `L`), and leaving it in the residual would show up
as scatter in every fold. Components below two cycles are flagged, not dropped.
"""
import numpy as np
from astropy.timeseries import LombScargle
from scipy.ndimage import median_filter
from scipy.optimize import minimize_scalar

FREQUENCY_OVERSAMPLE = 10
DEFAULT_HARMONICS = 3
DEFAULT_MAX_COMPONENTS = 6
# Breger+ 1993: below amplitude signal-to-noise 4 a peak in the amplitude
# spectrum is not distinguishable from the noise realisation.
DEFAULT_SNR_MIN = 4.0
# Half-width of the window used to measure the local noise of the amplitude
# spectrum, in cycles/day. Wide enough to average many independent frequencies,
# narrow enough that the strong red-noise rise at low frequency is local.
NOISE_WINDOW_CPD = 1.0
# Two frequencies whose ratio is within this of a small integer ratio belong to
# the same variation (a harmonic series), not to two independent ones.
COMMENSURATE_TOLERANCE = 0.03
COMMENSURATE_MAX_ORDER = 4


def to_relative(flux):
    """Flux -> (relative flux in ppt, mean). The inverse is `from_relative`."""
    mean_flux = float(np.mean(flux))
    return (np.asarray(flux, float) / mean_flux - 1.0) * 1e3, mean_flux


def from_relative(relative_flux, mean_flux):
    return mean_flux * (1.0 + np.asarray(relative_flux, float) / 1e3)


def frequency_grid(time, oversample=FREQUENCY_OVERSAMPLE):
    """Grid from 1/baseline to the Nyquist frequency of the median cadence."""
    time = np.asarray(time, float)
    baseline = time.max() - time.min()
    cadence = float(np.median(np.diff(np.sort(time))))
    min_frequency = 1.0 / baseline
    max_frequency = 1.0 / (2.0 * cadence)
    step = 1.0 / (oversample * baseline)
    return np.arange(min_frequency, max_frequency + step, step)


def amplitude_spectrum(time, flux, frequencies):
    """Semi-amplitude spectrum, same units as `flux`.

    `normalization="psd"` returns N/4 * amplitude^2 for a sinusoid, so the
    semi-amplitude is 2*sqrt(power/N). Using the amplitude and not the power is
    what makes the signal-to-noise below comparable to the literature.
    """
    power = LombScargle(time, flux).power(np.asarray(frequencies, float),
                                          normalization="psd")
    return 2.0 * np.sqrt(np.abs(power) / len(np.asarray(time)))


def harmonic_design(time, frequency, n_harmonics):
    columns = [np.ones_like(time)]
    for order in range(1, n_harmonics + 1):
        phase = 2.0 * np.pi * order * frequency * np.asarray(time, float)
        columns.append(np.sin(phase))
        columns.append(np.cos(phase))
    return np.column_stack(columns)


def fit_harmonic_model(time, flux, frequency, n_harmonics=DEFAULT_HARMONICS):
    """Least squares fit of an offset plus `n_harmonics` harmonics of `frequency`.

    Returns the zero-mean model and the semi-amplitude of each order. The
    harmonics are part of the same variation: an eclipse or an ellipsoidal
    modulation is not a sinusoid, and fitting only the fundamental would leave
    its own shape behind in the residual as a spurious extra component.
    """
    design = harmonic_design(time, frequency, n_harmonics)
    coefficients, *_ = np.linalg.lstsq(design, np.asarray(flux, float), rcond=None)
    model = design[:, 1:] @ coefficients[1:]
    sine = coefficients[1::2]
    cosine = coefficients[2::2]
    return model, np.hypot(sine, cosine)


def refine_frequency(time, flux, frequency, n_harmonics=DEFAULT_HARMONICS,
                     search_half_width=None):
    """Frequency that minimises the residual of the harmonic fit.

    The grid peak is only accurate to a fraction of the Rayleigh resolution;
    over a 27-day sector that error is enough to smear a fold by a tenth of a
    cycle. The search is bounded to one Rayleigh width so it refines the peak
    it was given instead of walking to a different one.
    """
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    rayleigh = 1.0 / (time.max() - time.min())
    half_width = search_half_width if search_half_width is not None else rayleigh

    def residual_sum_of_squares(trial_frequency):
        model, _ = fit_harmonic_model(time, flux, trial_frequency, n_harmonics)
        return float(np.sum((flux - np.mean(flux) - model) ** 2))

    result = minimize_scalar(
        residual_sum_of_squares,
        bounds=(max(frequency - half_width, 1e-6), frequency + half_width),
        method="bounded", options={"xatol": rayleigh * 1e-4})
    return float(result.x) if result.success else float(frequency)


def noise_spectrum(frequencies, amplitudes, window=NOISE_WINDOW_CPD):
    """`local_noise` evaluated everywhere: the running median of the spectrum.

    Times `snr_min` it is the curve a peak has to clear to be extracted, which
    is what makes the stopping criterion visible instead of a number.
    """
    frequencies = np.asarray(frequencies, float)
    step = float(np.median(np.diff(frequencies)))
    size = max(int(round(2.0 * window / step)), 3)
    return median_filter(np.asarray(amplitudes, float), size=size, mode="nearest")


def local_noise(frequencies, amplitudes, frequency, window=NOISE_WINDOW_CPD,
                statistic="median"):
    """Amplitude of the spectrum within `window` of `frequency`.

    Breger+ 1993 specifies the MEAN of the window. The median is used by
    default because it is not dragged up by real peaks that survive inside the
    window — in a star with a frequency group there is no clean stretch of pure
    noise within 1 c/d, and the mean there measures the signal, not the floor.
    For a pure Rayleigh noise realisation the median runs 0.939 of the mean, so
    the median gives a signal-to-noise ~6.5% higher. `statistic="mean"`
    restores the literal Breger convention.
    """
    near = np.abs(np.asarray(frequencies) - frequency) <= window
    if near.sum() < 10:
        near = np.ones(len(frequencies), dtype=bool)
    values = np.asarray(amplitudes)[near]
    return float(np.mean(values) if statistic == "mean" else np.median(values))


def extract_components(time, flux, n_max=DEFAULT_MAX_COMPONENTS,
                       n_harmonics=DEFAULT_HARMONICS, snr_min=DEFAULT_SNR_MIN,
                       oversample=FREQUENCY_OVERSAMPLE, return_history=False):
    """Prewhiten `flux` (ppt) one frequency at a time.

    At every step the strongest peak of the amplitude spectrum of the current
    residual is refined, fitted with its harmonics and subtracted. Stops when
    the peak no longer reaches `snr_min` over the local noise of that same
    residual, so the criterion tightens as the star is emptied of signal.

    Returns (components, residual), or (components, residual, history) with
    `return_history`: one entry per iteration with the residual and the spectrum
    it was chosen on, including the rejected last one. Each component carries
    the model evaluated on `time`, so any subset can be added back or removed
    later.
    """
    time = np.asarray(time, float)
    residual = np.asarray(flux, float) - np.mean(flux)
    grid = frequency_grid(time, oversample)
    baseline = time.max() - time.min()

    components = []
    history = []
    for index in range(n_max):
        spectrum = amplitude_spectrum(time, residual, grid)
        peak = int(np.argmax(spectrum))
        frequency = refine_frequency(time, residual, grid[peak], n_harmonics)
        model, amplitudes = fit_harmonic_model(time, residual, frequency,
                                               n_harmonics)
        noise = local_noise(grid, spectrum, frequency)
        signal_to_noise = float(amplitudes[0] / noise) if noise > 0 else np.inf
        if return_history:
            history.append({
                "step": index + 1, "residual": residual.copy(),
                "spectrum": spectrum, "frequency": frequency,
                "amplitude": float(amplitudes[0]), "snr": signal_to_noise,
                "model": model, "accepted": signal_to_noise >= snr_min,
            })
        if signal_to_noise < snr_min:
            break
        residual = residual - model
        components.append({
            "index": index + 1,
            "frequency": frequency,
            "period": 1.0 / frequency,
            "amplitude": float(amplitudes[0]),
            "amplitude_harmonics": amplitudes[1:].astype(float),
            "snr": signal_to_noise,
            "cycles": float(baseline * frequency),
            "model": model,
        })
    if return_history:
        return components, residual, history
    return components, residual


def commensurate(first, second, max_order=COMMENSURATE_MAX_ORDER,
                 tolerance=COMMENSURATE_TOLERANCE):
    """True if the two frequencies are members of one harmonic series.

    Checked as p/q with p, q <= max_order and a RELATIVE tolerance on the
    ratio: an absolute tolerance on a ratio of 2 is a five times tighter test
    than on a ratio of 1, which is what made the harmonic count in
    `match_periods.py` collapse before it was made relative.
    """
    ratio = float(first) / float(second)
    for numerator in range(1, max_order + 1):
        for denominator in range(1, max_order + 1):
            target = numerator / denominator
            if abs(ratio / target - 1.0) < tolerance:
                return True
    return False


def isolate(flux, components, target_frequency, keep_commensurate=True):
    """`flux` with every component that is not the target variation removed.

    A component commensurate with the target is part of the SAME variation —
    the first harmonic of an ellipsoidal is what gives it two maxima per orbit
    — so removing it would erase the very shape the fold is meant to show. Set
    `keep_commensurate=False` to force a bare sinusoid at the target instead.
    """
    cleaned = np.asarray(flux, float).copy()
    for component in components:
        if keep_commensurate and commensurate(component["frequency"],
                                              target_frequency):
            continue
        cleaned = cleaned - component["model"]
    return cleaned


def isolate_single(flux, components, index):
    """Only the component at `index`, every other one removed.

    Stricter than `isolate`: it drops the commensurate ones too, so a harmonic
    group is not carried along with its fundamental. That is the wrong curve to
    judge an ellipsoidal by, and the right one to answer "what does THIS
    frequency look like on its own".
    """
    cleaned = np.asarray(flux, float).copy()
    for position, component in enumerate(components):
        if position != index:
            cleaned = cleaned - component["model"]
    return cleaned


def probe(time, flux, components, frequency, n_harmonics=DEFAULT_HARMONICS,
          oversample=FREQUENCY_OVERSAMPLE):
    """Amplitude and signal-to-noise of `flux` at one imposed frequency.

    This is the test for a published period that the pipeline never found: fit
    a sinusoid exactly there, on the light curve with the other variations
    removed, and read off whether anything is present. An amplitude far below
    the local noise means the period is not in the data — as opposed to being
    in the data but invisible in the fold.
    """
    time = np.asarray(time, float)
    cleaned = isolate(flux, components, frequency)
    cleaned = cleaned - np.mean(cleaned)
    model, amplitudes = fit_harmonic_model(time, cleaned, frequency, n_harmonics)
    grid = frequency_grid(time, oversample)
    spectrum = amplitude_spectrum(time, cleaned - model, grid)
    noise = local_noise(grid, spectrum, frequency)
    noise_mean = local_noise(grid, spectrum, frequency, statistic="mean")
    return {
        "frequency": float(frequency),
        "period": 1.0 / float(frequency),
        "amplitude": float(amplitudes[0]),
        "amplitude_harmonics": amplitudes[1:].astype(float),
        "noise": noise,
        "noise_mean": noise_mean,
        "snr": float(amplitudes[0] / noise) if noise > 0 else np.inf,
        "snr_breger": (float(amplitudes[0] / noise_mean)
                       if noise_mean > 0 else np.inf),
        # `snr` mide SOLO la sinusoide fundamental, y eso penaliza a las
        # eclipsantes separadas por su forma y no por su falta de señal: una
        # curva plana con eclipses angostos reparte su potencia entre muchos
        # armónicos y deja a1 pequeña. La suma en cuadratura del modelo
        # armónico completo mide "cuánta señal coherente hay EN este período",
        # que es la pregunta que el veto quiere hacer.
        "amplitude_model": float(np.sqrt(np.sum(np.square(amplitudes)))),
        "snr_model": (float(np.sqrt(np.sum(np.square(amplitudes))) / noise)
                      if noise > 0 else np.inf),
        "cycles": float((time.max() - time.min()) * frequency),
        "isolated": cleaned,
        "model": model,
    }


def cluster_components(components, resolution):
    """Label components closer than `resolution` in frequency as one group.

    Two frequencies separated by less than the Rayleigh resolution 1/T are not
    resolved: prewhitening still splits them, but what is there is ONE broad
    structure and not two periods. A cluster of several unresolved components
    is exactly what Labadie-Bartz+ 2022 call a frequency group, and it is why
    the fold at the centre of one does not close.
    """
    order = sorted(range(len(components)),
                   key=lambda position: components[position]["frequency"])
    labels = [0] * len(components)
    label = 0
    for rank, position in enumerate(order):
        if rank > 0:
            previous = components[order[rank - 1]]["frequency"]
            if components[position]["frequency"] - previous > resolution:
                label += 1
        labels[position] = label
    return labels
