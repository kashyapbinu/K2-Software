"""
K2 Aerospace — Vibration Analysis
====================================
Frequency response, Miles' equation, random vibration PSD response.

Physics formulations
--------------------
- Complex FRF via modal superposition:
      H(ω) = Σ_r (1/m_r) / (ω_r² − ω² + j·2·ζ_r·ω_r·ω)
  Ref: Bisplinghoff, Ashley & Halfman, "Aeroelasticity", §5.4;
       Craig & Kurdila, "Fundamentals of Structural Dynamics", Ch. 5.

- CQC (Complete Quadratic Combination) per Der Kiureghian & Nakamura (1993):
      R² = Σ_i Σ_j ρ_ij · R_i · R_j
  with correlation coefficient:
      ρ_ij = 8·√(ζ_i·ζ_j)·(ζ_i + r·ζ_j)·r^(3/2)
             / ((1−r²)² + 4·ζ_i·ζ_j·r·(1+r²) + 4·(ζ_i²+ζ_j²)·r²)
  where r = ω_j/ω_i.

- Miles' equation (unchanged):
      G_rms = √(π/2 · f_n · Q · W)
  Ref: NASA-STD-7001, Miles (1954).

- Displacement PSD from acceleration FRF:
      |H_disp(f)|² = |H_accel(f)|² / ω⁴
  Integrated numerically via trapezoidal rule.
"""
from __future__ import annotations
import cmath
import math
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("K2.Dynamics.Vibration")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class VibrationResult:
    """Results from vibration analysis.

    All original fields are preserved for backward compatibility.
    New fields carry defaults so old call-sites continue to work.
    """
    # Natural frequencies used
    natural_frequencies_hz: list = field(default_factory=list)

    # FRF data: [(freq_hz, magnitude_dB, phase_deg), ...]
    frf_data: list = field(default_factory=list)

    # RMS responses
    rms_acceleration_g: float = 0.0        # g-rms
    rms_displacement_mm: float = 0.0       # mm-rms (broadband integrated)
    peak_response_g: float = 0.0           # 3-sigma peak acceleration (g)

    # PSD response: [(freq_hz, psd_g2_hz), ...]
    response_psd: list = field(default_factory=list)

    # Input PSD level
    input_psd_g2_hz: float = 0.0           # g²/Hz

    # --- NEW fields -----------------------------------------------------------
    # Modal markers: peaks in combined FRF matched to eigenfrequencies
    # [(freq_hz, magnitude_dB, mode_name), ...]
    modal_markers: list = field(default_factory=list)

    # Whether every identified FRF peak lies within ±5 % of an eigenfrequency
    peaks_validated: bool = False

    # Per-mode FRF contributions: list of [(freq_hz, magnitude_dB), ...] per mode
    mode_contributions: list = field(default_factory=list)

    # Relative participation of each mode (fraction of total RMS²)
    mode_participation: list = field(default_factory=list)

    # CQC-combined RMS (g)  — for comparison with SRSS
    rms_acceleration_cqc_g: float = 0.0    # g-rms via CQC

    # Damping ratios used per mode
    damping_ratios_used: list = field(default_factory=list)

    # Modal masses used per mode
    modal_masses_used: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Utility — CQC correlation coefficient
# ---------------------------------------------------------------------------
def _cqc_rho(zeta_i: float, zeta_j: float, r: float) -> float:
    """Der Kiureghian cross-modal correlation coefficient.

    ρ_ij = 8·√(ζ_i·ζ_j)·(ζ_i + r·ζ_j)·r^(3/2)
           / ((1−r²)² + 4·ζ_i·ζ_j·r·(1+r²) + 4·(ζ_i²+ζ_j²)·r²)

    Parameters
    ----------
    zeta_i, zeta_j : float   Modal damping ratios.
    r : float                Frequency ratio ω_j / ω_i.

    Returns
    -------
    float : Correlation coefficient ρ ∈ [0, 1].

    Reference
    ---------
    Der Kiureghian, A. & Nakamura, Y. (1993).
    "CQC modal combination rule for high-frequency modes."
    Earthquake Engineering & Structural Dynamics, 22(11), 943-956.
    """
    if r <= 0.0:
        return 0.0
    num = 8.0 * math.sqrt(zeta_i * zeta_j) * (zeta_i + r * zeta_j) * r ** 1.5
    d1 = (1.0 - r ** 2) ** 2
    d2 = 4.0 * zeta_i * zeta_j * r * (1.0 + r ** 2)
    d3 = 4.0 * (zeta_i ** 2 + zeta_j ** 2) * r ** 2
    denom = d1 + d2 + d3
    if denom <= 0.0:
        return 1.0
    return min(num / denom, 1.0)


# ---------------------------------------------------------------------------
# Utility — detect local peaks in a sampled magnitude curve
# ---------------------------------------------------------------------------
def _find_peaks_db(freq_mag: list) -> list:
    """Find local maxima in a (freq, mag_dB) list.

    Returns list of (index, freq_hz, mag_dB).
    """
    peaks = []
    n = len(freq_mag)
    if n < 3:
        return peaks
    for k in range(1, n - 1):
        f_prev, m_prev = freq_mag[k - 1]
        f_curr, m_curr = freq_mag[k]
        f_next, m_next = freq_mag[k + 1]
        if m_curr > m_prev and m_curr > m_next:
            peaks.append((k, f_curr, m_curr))
    return peaks


# ---------------------------------------------------------------------------
# Public: single-DOF FRF (unchanged signature)
# ---------------------------------------------------------------------------
def frequency_response(natural_freq_hz: float, damping_ratio: float = 0.02,
                       freq_range: tuple = (1, 2000), n_points: int = 500) -> list:
    """Compute FRF (H(f)) for a single-DOF system.

    Returns list of (freq_hz, magnitude_dB, phase_deg).
    H(f) = 1 / sqrt((1 − r²)² + (2·ζ·r)²)  where r = f/fn

    Reference
    ---------
    Thomson, "Theory of Vibration with Applications", §3.6.
    """
    f_start, f_end = freq_range
    result = []
    for i in range(n_points):
        f = f_start * (f_end / f_start) ** (i / (n_points - 1))  # log spacing
        r = f / natural_freq_hz if natural_freq_hz > 0 else 0.0
        denom = math.sqrt((1.0 - r ** 2) ** 2 + (2.0 * damping_ratio * r) ** 2)
        mag = 1.0 / max(denom, 1e-12)
        mag_db = 20.0 * math.log10(max(mag, 1e-12))
        phase = -math.degrees(math.atan2(2.0 * damping_ratio * r, 1.0 - r ** 2))
        result.append((f, mag_db, phase))
    return result


# ---------------------------------------------------------------------------
# Public: Miles' equation (unchanged signature)
# ---------------------------------------------------------------------------
def miles_equation(fn: float, damping_ratio: float, input_psd_g2_hz: float) -> float:
    """Miles' equation: RMS response to broadband random vibration.

    G_rms = sqrt(π/2 × fn × Q × W)
    where Q = 1/(2ζ), W = input PSD (g²/Hz)

    Parameters
    ----------
    fn : float             Natural frequency (Hz)
    damping_ratio : float  Damping ratio ζ
    input_psd_g2_hz : float  Input acceleration PSD (g²/Hz)

    Returns
    -------
    float : RMS acceleration response (g)

    Reference
    ---------
    Miles, J.W. (1954). "On Structural Fatigue Under Random Loading."
    Journal of the Aeronautical Sciences, 21(11), 753-762.
    """
    if fn <= 0 or damping_ratio <= 0:
        return 0.0
    Q = 1.0 / (2.0 * damping_ratio)
    grms = math.sqrt(math.pi / 2.0 * fn * Q * input_psd_g2_hz)
    return grms


# ---------------------------------------------------------------------------
# Public: full random vibration response (extended, backward-compatible)
# ---------------------------------------------------------------------------
def random_vibration_response(natural_freqs: list, damping_ratio: float = 0.02,
                               input_psd_g2_hz: float = 0.04,
                               freq_range: tuple = (20, 2000),
                               n_points: int = 400,
                               damping_ratios: list = None,
                               modal_masses: list = None,
                               mode_names: list = None) -> VibrationResult:
    """Full random vibration analysis with complex modal superposition.

    Computes:
    1. Complex FRF via modal superposition  (proper phase retained)
    2. Per-mode FRF contributions
    3. CQC modal combination for RMS response
    4. Modal markers on FRF peaks with eigenfrequency validation
    5. Broadband displacement via spectral integration of H_disp

    NASA-STD-7001 typical input: 0.04 g²/Hz from 20–2000 Hz.

    Parameters
    ----------
    natural_freqs : list[float]
        Natural frequencies (Hz).
    damping_ratio : float
        Scalar modal damping ratio ζ  (used when *damping_ratios* is None).
    input_psd_g2_hz : float
        Input acceleration PSD level (g²/Hz), assumed flat.
    freq_range : tuple
        Analysis frequency range (Hz).
    n_points : int
        Number of log-spaced frequency points.
    damping_ratios : list[float] | None
        Per-mode damping ratios.  Falls back to scalar *damping_ratio*.
    modal_masses : list[float] | None
        Per-mode generalised masses (kg).  Defaults to 1.0 for each mode.
    mode_names : list[str] | None
        Human-readable labels, e.g. ["Mode 1 — 1st bend", ...].

    Returns
    -------
    VibrationResult

    References
    ----------
    Bisplinghoff, Ashley & Halfman, "Aeroelasticity", §5.4.
    Craig & Kurdila, "Fundamentals of Structural Dynamics", Ch. 5.
    Der Kiureghian & Nakamura (1993), Earthquake Eng. & Struct. Dyn.
    """
    # ------------------------------------------------------------------
    # 0.  Sanitise inputs & build per-mode arrays
    # ------------------------------------------------------------------
    n_modes = len(natural_freqs)
    result = VibrationResult()
    result.natural_frequencies_hz = list(natural_freqs)
    result.input_psd_g2_hz = input_psd_g2_hz

    # Per-mode damping
    if damping_ratios is not None and len(damping_ratios) == n_modes:
        zetas = list(damping_ratios)
    else:
        zetas = [damping_ratio] * n_modes
    result.damping_ratios_used = list(zetas)

    # Per-mode generalised mass
    if modal_masses is not None and len(modal_masses) == n_modes:
        m_r = list(modal_masses)
    else:
        m_r = [1.0] * n_modes
    result.modal_masses_used = list(m_r)

    # Mode labels
    if mode_names is not None and len(mode_names) == n_modes:
        names = list(mode_names)
    else:
        names = [f"Mode {k+1}" for k in range(n_modes)]

    # Pre-compute angular eigenfrequencies
    omega_r = [TWO_PI * fn for fn in natural_freqs]  # rad/s

    # ------------------------------------------------------------------
    # 1.  Complex FRF via modal superposition  &  per-mode contributions
    # ------------------------------------------------------------------
    f_start, f_end = freq_range
    combined_frf = []          # (freq_hz, mag_dB, phase_deg)
    response_psd = []          # (freq_hz, psd_g²/Hz)
    mode_contribs = [[] for _ in range(n_modes)]  # per-mode FRF curves
    disp_psd_curve = []        # (freq_hz, psd_mm²/Hz) for displacement

    for i in range(n_points):
        f = f_start * (f_end / f_start) ** (i / (n_points - 1))  # log spacing
        omega = TWO_PI * f

        # Complex transfer function: H(ω) = Σ_r (1/m_r) / (ω_r² − ω² + j·2·ζ_r·ω_r·ω)
        H_total = complex(0.0, 0.0)
        for r_idx in range(n_modes):
            if natural_freqs[r_idx] <= 0.0:
                continue
            wr = omega_r[r_idx]
            zr = zetas[r_idx]
            mr = m_r[r_idx]
            denom_c = complex(wr ** 2 - omega ** 2, 2.0 * zr * wr * omega)
            if abs(denom_c) < 1e-30:
                denom_c = complex(1e-30, 0.0)
            H_mode = (1.0 / mr) / denom_c
            H_total += H_mode

            # Per-mode contribution magnitude (dB)
            mag_mode = abs(H_mode)
            mag_mode_db = 20.0 * math.log10(max(mag_mode, 1e-30))
            mode_contribs[r_idx].append((f, mag_mode_db))

        # Combined magnitude & phase
        mag_total = abs(H_total)
        mag_total_db = 20.0 * math.log10(max(mag_total, 1e-30))
        phase_total_deg = math.degrees(cmath.phase(H_total))
        combined_frf.append((f, mag_total_db, phase_total_deg))

        # Output PSD = |H(f)|² × W_input
        H_sq = mag_total ** 2
        out_psd = H_sq * input_psd_g2_hz
        response_psd.append((f, out_psd))

        # Displacement PSD: |H_disp(f)|² = |H_accel(f)|² / ω⁴
        # Convert accel g²/Hz → (m/s²)²/Hz, divide by ω⁴, → m², then to mm²
        if omega > 0.0:
            # accel PSD in (m/s²)²/Hz
            accel_psd_si = out_psd * 9.81 ** 2
            disp_psd_si = accel_psd_si / omega ** 4   # m²/Hz
            disp_psd_mm2 = disp_psd_si * 1e6          # mm²/Hz
        else:
            disp_psd_mm2 = 0.0
        disp_psd_curve.append((f, disp_psd_mm2))

    result.frf_data = combined_frf
    result.response_psd = response_psd
    result.mode_contributions = mode_contribs

    # ------------------------------------------------------------------
    # 2.  CQC-combined RMS acceleration  (replaces SRSS)
    # ------------------------------------------------------------------
    # Per-mode RMS via Miles' equation
    grms_per_mode = []
    for r_idx in range(n_modes):
        fn = natural_freqs[r_idx]
        zr = zetas[r_idx]
        grms_per_mode.append(miles_equation(fn, zr, input_psd_g2_hz))

    # CQC: R² = Σ_i Σ_j ρ_ij · R_i · R_j
    rms_sq_cqc = 0.0
    for i_m in range(n_modes):
        for j_m in range(n_modes):
            wi = omega_r[i_m] if natural_freqs[i_m] > 0 else 0.0
            wj = omega_r[j_m] if natural_freqs[j_m] > 0 else 0.0
            if wi > 0.0:
                r_ratio = wj / wi
            else:
                r_ratio = 0.0
            rho = _cqc_rho(zetas[i_m], zetas[j_m], r_ratio)
            rms_sq_cqc += rho * grms_per_mode[i_m] * grms_per_mode[j_m]

    result.rms_acceleration_cqc_g = math.sqrt(max(rms_sq_cqc, 0.0))

    # Primary RMS uses CQC (more accurate for closely-spaced modes)
    result.rms_acceleration_g = result.rms_acceleration_cqc_g
    result.peak_response_g = result.rms_acceleration_g * 3.0  # 3σ peak

    # Mode participation: fraction of total RMS² contributed by each mode
    total_grms_sq = sum(g ** 2 for g in grms_per_mode)
    if total_grms_sq > 0.0:
        result.mode_participation = [g ** 2 / total_grms_sq for g in grms_per_mode]
    else:
        result.mode_participation = [0.0] * n_modes

    # ------------------------------------------------------------------
    # 3.  Broadband displacement via trapezoidal integration of disp PSD
    # ------------------------------------------------------------------
    disp_rms_sq = 0.0
    for k in range(1, len(disp_psd_curve)):
        f0, s0 = disp_psd_curve[k - 1]
        f1, s1 = disp_psd_curve[k]
        df = f1 - f0
        disp_rms_sq += 0.5 * (s0 + s1) * df  # trapezoidal rule, mm²
    result.rms_displacement_mm = math.sqrt(max(disp_rms_sq, 0.0))

    # ------------------------------------------------------------------
    # 4.  Modal markers — identify FRF peaks & match to eigenfrequencies
    # ------------------------------------------------------------------
    freq_mag_pairs = [(f, m) for (f, m, _) in combined_frf]
    raw_peaks = _find_peaks_db(freq_mag_pairs)

    markers = []
    all_matched = True
    tolerance = 0.05  # ±5 % frequency tolerance

    for _, fp, mp in raw_peaks:
        matched_name = None
        for r_idx, fn in enumerate(natural_freqs):
            if fn <= 0.0:
                continue
            if abs(fp - fn) / fn <= tolerance:
                matched_name = names[r_idx]
                break
        if matched_name is None:
            all_matched = False
            matched_name = f"Unmatched ({fp:.1f} Hz)"
        markers.append((fp, mp, matched_name))

    result.modal_markers = markers
    result.peaks_validated = all_matched and len(markers) > 0

    # ------------------------------------------------------------------
    # 5.  Logging
    # ------------------------------------------------------------------
    logger.info(
        f"Vibration: G_rms={result.rms_acceleration_g:.2f}g (CQC), "
        f"G_peak(3σ)={result.peak_response_g:.2f}g, "
        f"δ_rms={result.rms_displacement_mm:.3f}mm, "
        f"peaks_validated={result.peaks_validated}"
    )
    return result
