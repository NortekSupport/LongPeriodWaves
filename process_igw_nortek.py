# process_igw_nortek.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, welch
import pywt


@dataclass
class SpectralResults:
    Hm0_hour: Optional[np.ndarray] = None
    Tm01_hour: Optional[np.ndarray] = None
    Tm02_hour: Optional[np.ndarray] = None
    Tp_hour: Optional[np.ndarray] = None
    freq: Optional[np.ndarray] = None  # shape (F, n_segments)
    SB: Optional[np.ndarray] = None    # PSD [units^2/Hz], shape (F, n_segments)


@dataclass
class ZDCResults:
    filtered_TS: Optional[np.ndarray] = None
    filtered_SS: Optional[np.ndarray] = None
    Hm0: Optional[np.ndarray] = None
    Tm01: Optional[np.ndarray] = None
    Tm02: Optional[np.ndarray] = None
    Tp: Optional[np.ndarray] = None
    f: Optional[np.ndarray] = None      # per-segment frequency arrays (n_seg, F) or (F,)
    SB: Optional[np.ndarray] = None     # per-segment IG-band spectra (n_seg, F)
    Flag_hour: Optional[np.ndarray] = None  # boolean per hour


@dataclass
class WaveletResults:
    f: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    pow: Optional[np.ndarray] = None        # power (units^2), shape (F, N)
    pow_frac: Optional[np.ndarray] = None   # fraction of variance, shape (F, N)
    pow_dB: Optional[np.ndarray] = None     # 10*log10(pow)


@dataclass
class SensorGroup:
    Spectral: SpectralResults = field(default_factory=SpectralResults)
    zdc: ZDCResults = field(default_factory=ZDCResults)
    Wavelet: WaveletResults = field(default_factory=WaveletResults)
    time_hourly: Optional[np.ndarray] = None


@dataclass
class IGGroup:
    Spectral: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    zdc: Dict[str, ZDCResults] = field(default_factory=dict)
    Wavelet: Dict[str, WaveletResults] = field(default_factory=dict)
    Time_hour: Optional[np.ndarray] = None
    time: Optional[np.ndarray] = None


def process_igw_nortek(
    time: np.ndarray,
    Fs: float,
    *,
    pressure: Optional[np.ndarray] = None,
    ast: Optional[np.ndarray] = None,
    IG_band: Tuple[float, float] = (1 / 250.0, 1 / 25.0),
    seaswell_band: Tuple[float, float] = (1 / 25.0, 1 / 5.0),
    rho: float = 1025.0,
    g: float = 9.81,
    p2eta_fn: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    welch_nperseg: int = 2**12,
    welch_overlap: float = 0.5,
    wavelet_freq_limits: Tuple[float, float] = (1 / 300.0, 1 / 5.0),
    wavelet_voices_per_octave: int = 10,
    wavelet_seg_sec: int = 600,
    wavelet_hop_sec: int = 300,
) -> Tuple[IGGroup, SensorGroup, SensorGroup]:
    """
    Python translation of ProcessIGW_Nortek.

    Parameters
    ----------
    time : array (N,)
        Time vector [s] or datetime64 (monotonic). Only used to derive hourly ticks.
    Fs : float
        Sampling frequency [Hz].
    pressure : array, optional
        Pressure time series [Pa].
    ast : array, optional
        AST elevation time series [m].
    IG_band : (f_low, f_high)
        Infragravity frequency bounds [Hz].
    seaswell_band : (f_low, f_high)
        Sea/Swell band [Hz] (used for zdc SS filtering).
    rho, g : floats
        Water density [kg/m^3] and gravity [m/s^2] for hydrostatic conversion.
    p2eta_fn : callable, optional
        If provided, converts pressure segment -> surface elevation in meters.
        Signature: eta = p2eta_fn(pressure_segment, Fs)
        If None, a simple hydrostatic conversion is used: (p - mean(p)) / (rho*g).
    welch_nperseg, welch_overlap : int, float
        Welch PSD parameters.
    wavelet_* : wavelet analysis parameters.

    Returns
    -------
    IG : IGGroup
    Pressure : SensorGroup
    AST : SensorGroup
    """
    if pressure is None and ast is None:
        raise ValueError("Provide at least one sensor: pressure and/or ast.")

    samplesInHour = int(np.floor(3600 * Fs))

    IG = IGGroup()
    Pressure = SensorGroup()
    AST = SensorGroup()

    IG.Time_hour = time[::samplesInHour] if time is not None and len(time) else None
    IG.time = time

    # ----------------- helpers -----------------
    def _hourly_segments(x: np.ndarray) -> int:
        return int(np.floor(len(x) / samplesInHour))

    def _welch_psd(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        nperseg = min(welch_nperseg, len(x))
        noverlap = int(welch_overlap * nperseg)
        f, Pxx = welch(
            x,
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap if noverlap < nperseg else nperseg // 2,
            detrend="constant",
            return_onesided=True,
            scaling="density",
            window="hann",
        )
        return f, Pxx

    def _band_moments(f: np.ndarray, S: np.ndarray, band: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """Return Hm0, Tm01, Tm02, Tp for a band using discrete integration."""
        mask = (f >= band[0]) & (f <= band[1])
        if not np.any(mask):
            return np.nan, np.nan, np.nan, np.nan
        fi = f[mask]
        Si = S[mask]
        if fi.size < 2:
            return np.nan, np.nan, np.nan, np.nan
        df = np.mean(np.diff(fi))
        m0 = np.nansum(Si) * df
        m1 = np.nansum(fi * Si) * df
        m2 = np.nansum((fi**2) * Si) * df
        Hm0 = 4.0 * np.sqrt(m0) if m0 > 0 else np.nan
        Tm01 = m0 / m1 if (m0 > 0 and m1 > 0) else np.nan
        Tm02 = np.sqrt(m0 / m2) if (m0 > 0 and m2 > 0) else np.nan
        Tp = 1.0 / fi[np.nanargmax(Si)] if np.all(np.isfinite(fi)) else np.nan
        return Hm0, Tm01, Tm02, Tp

    def _design_bp(f1: float, f2: float, fs: float, order: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        wn = [f1 / (fs / 2.0), f2 / (fs / 2.0)]
        b, a = butter(order, wn, btype="bandpass", output="ba")
        return b, a

    def _zdc_segment_metrics(seg: np.ndarray, fs: float) -> Tuple[float, float]:
        # zero down-crossings (from >0 to <=0)
        s1 = seg[:-1]
        s2 = seg[1:]
        crossings = np.where((s1 > 0) & (s2 <= 0))[0]
        nW = max(0, len(crossings) - 1)
        if nW < 1:
            return np.nan, np.nan
        H = np.empty(nW)
        T = np.empty(nW)
        for j in range(nW):
            w = seg[crossings[j] : crossings[j + 1] + 1]
            H[j] = np.nanmax(w) - np.nanmin(w)
            T[j] = (crossings[j + 1] - crossings[j]) / fs
        return np.nanmax(H), np.nanmax(T)

    def _pressure_to_eta(pseg: np.ndarray, fs: float) -> np.ndarray:
        if p2eta_fn is not None:
            return p2eta_fn(pseg, fs)
        # hydrostatic approximation, remove mean
        return (pseg - np.nanmean(pseg)) / (rho * g)

    # ----------------- spectral analysis -----------------
    def _run_spectral(pressure_raw: Optional[np.ndarray],
                      ast_raw: Optional[np.ndarray]) -> Tuple[IGGroup, SensorGroup, SensorGroup]:

        # AST branch
        if ast_raw is not None:
            nsegA = _hourly_segments(ast_raw)
            if nsegA > 0:
                freqs = []
                psds = []
                Hm0_lst, Tm01_lst, Tm02_lst, Tp_lst = [], [], [], []
                for i in range(nsegA):
                    idx = slice(i * samplesInHour, (i + 1) * samplesInHour)
                    seg = ast_raw[idx]
                    f, S = _welch_psd(seg - np.nanmean(seg), Fs)
                    freqs.append(f)
                    psds.append(S)
                    Hm0, Tm01, Tm02, Tp = _band_moments(f, S, IG_band)
                    Hm0_lst.append(Hm0)
                    Tm01_lst.append(Tm01)
                    Tm02_lst.append(Tm02)
                    Tp_lst.append(Tp)

                AST.Spectral.Hm0_hour = np.asarray(Hm0_lst)
                AST.Spectral.Tm01_hour = np.asarray(Tm01_lst)
                AST.Spectral.Tm02_hour = np.asarray(Tm02_lst)
                AST.Spectral.Tp_hour = np.asarray(Tp_lst)
                # pack as (F, nseg)
                Fmax = max(len(f) for f in freqs)
                Fgrid = freqs[0]
                SB = np.vstack([np.interp(Fgrid, freqs[i], psds[i]) for i in range(nsegA)]).T
                AST.Spectral.freq = np.tile(Fgrid[:, None], (1, nsegA))
                AST.Spectral.SB = SB

                # IG-band (AST) moments also mirrored in IG namespace
                IG.Spectral.setdefault("AST", {})
                IG.Spectral["AST"]["Hm0"] = AST.Spectral.Hm0_hour
                IG.Spectral["AST"]["Tm01"] = AST.Spectral.Tm01_hour
                IG.Spectral["AST"]["Tm02"] = AST.Spectral.Tm02_hour
                IG.Spectral["AST"]["Tp"] = AST.Spectral.Tp_hour

        # Pressure branch
        if pressure_raw is not None:
            nsegP = _hourly_segments(pressure_raw)
            if nsegP > 0:
                freqs = []
                psds = []
                Hm0_lst, Tm01_lst, Tm02_lst, Tp_lst = [], [], [], []
                for i in range(nsegP):
                    idx = slice(i * samplesInHour, (i + 1) * samplesInHour)
                    pseg = pressure_raw[idx]
                    eta = _pressure_to_eta(pseg, Fs)
                    f, S = _welch_psd(eta - np.nanmean(eta), Fs)
                    freqs.append(f)
                    psds.append(S)
                    Hm0, Tm01, Tm02, Tp = _band_moments(f, S, IG_band)
                    Hm0_lst.append(Hm0)
                    Tm01_lst.append(Tm01)
                    Tm02_lst.append(Tm02)
                    Tp_lst.append(Tp)

                Pressure.Spectral.Hm0_hour = np.asarray(Hm0_lst)
                Pressure.Spectral.Tm01_hour = np.asarray(Tm01_lst)
                Pressure.Spectral.Tm02_hour = np.asarray(Tm02_lst)
                Pressure.Spectral.Tp_hour = np.asarray(Tp_lst)
                Fgrid = freqs[0]
                SB = np.vstack([np.interp(Fgrid, freqs[i], psds[i]) for i in range(nsegP)]).T
                Pressure.Spectral.freq = np.tile(Fgrid[:, None], (1, nsegP))
                Pressure.Spectral.SB = SB

                IG.Spectral.setdefault("Pressure", {})
                IG.Spectral["Pressure"]["Hm0"] = Pressure.Spectral.Hm0_hour
                IG.Spectral["Pressure"]["Tm01"] = Pressure.Spectral.Tm01_hour
                IG.Spectral["Pressure"]["Tm02"] = Pressure.Spectral.Tm02_hour
                IG.Spectral["Pressure"]["Tp"] = Pressure.Spectral.Tp_hour

        # hourly time vectors
        if time is not None and len(time):
            if Pressure.Spectral.SB is not None:
                nP = Pressure.Spectral.SB.shape[1]
                Pressure.time_hourly = time[::samplesInHour][:nP]
            if AST.Spectral.SB is not None:
                nA = AST.Spectral.SB.shape[1]
                AST.time_hourly = time[::samplesInHour][:nA]

        return IG, Pressure, AST

    # ----------------- ZDC analysis -----------------
    def _run_zdc(IG: IGGroup,
                 Pressure: SensorGroup,
                 AST: SensorGroup,
                 pressure_raw: Optional[np.ndarray],
                 ast_raw: Optional[np.ndarray]) -> Tuple[IGGroup, SensorGroup, SensorGroup]:

        # Filters
        bIG, aIG = _design_bp(IG_band[0], IG_band[1], Fs, order=6)
        bSS, aSS = _design_bp(seaswell_band[0], seaswell_band[1], Fs, order=6)

        # AST
        if ast_raw is not None:
            x = ast_raw - np.nanmean(ast_raw)
            IG_ts = filtfilt(bIG, aIG, x, method="gust")
            SS_ts = filtfilt(bSS, aSS, x, method="gust")
            AST.zdc.filtered_SS = SS_ts
            IG.zdc.setdefault("AST", ZDCResults())
            IG.zdc["AST"].filtered_TS = IG_ts

            nseg = _hourly_segments(ast_raw)
            Hm0 = np.full(nseg, np.nan)
            Tpx = np.full(nseg, np.nan)
            Tm01 = np.full(nseg, np.nan)
            Tm02 = np.full(nseg, np.nan)
            Flag = np.zeros(nseg, dtype=bool)
            fgrid_list = []
            SB_list = []

            for i in range(nseg):
                idx = slice(i * samplesInHour, (i + 1) * samplesInHour)
                seg = IG_ts[idx]

                # down-cross stats
                H, Tp_est = _zdc_segment_metrics(seg, Fs)
                if np.isfinite(H):
                    Hm0[i] = H
                if np.isfinite(Tp_est):
                    Tpx[i] = Tp_est
                # flag: at least ~50 waves counted
                s1 = seg[:-1]
                s2 = seg[1:]
                crossings = np.where((s1 > 0) & (s2 <= 0))[0]
                Flag[i] = (len(crossings) - 1) > 50

                # spectral stats on filtered IG segment
                f, S = _welch_psd(seg - np.nanmean(seg), Fs)
                Hm0_w, Tm01_w, Tm02_w, Tp_w = _band_moments(f, S, (1/250.0, 1/25.0))
                # store full-band for later (IG band in welch already integrated)
                fgrid_list.append(f)
                SB_list.append(S)
                if np.isfinite(Hm0_w): Hm0[i] = Hm0_w
                if np.isfinite(Tm01_w): Tm01[i] = Tm01_w
                if np.isfinite(Tm02_w): Tm02[i] = Tm02_w
                if np.isfinite(Tp_w):   Tpx[i]  = Tp_w

            IG.zdc["AST"].Hm0 = Hm0
            IG.zdc["AST"].Tm01 = Tm01
            IG.zdc["AST"].Tm02 = Tm02
            IG.zdc["AST"].Tp = Tpx
            IG.zdc["AST"].Flag_hour = Flag
            if fgrid_list:
                fgrid = fgrid_list[0]
                SB = np.vstack([np.interp(fgrid, fgrid_list[i], SB_list[i]) for i in range(nseg)])
                IG.zdc["AST"].f = fgrid
                IG.zdc["AST"].SB = SB

        # Pressure
        if pressure_raw is not None:
            p = pressure_raw - np.nanmean(pressure_raw)
            IG_ts = filtfilt(bIG, aIG, p, method="gust")
            SS_ts = filtfilt(bSS, aSS, p, method="gust")
            Pressure.zdc.filtered_SS = SS_ts
            IG.zdc.setdefault("Pressure", ZDCResults())
            IG.zdc["Pressure"].filtered_TS = IG_ts

            nseg = _hourly_segments(pressure_raw)
            Hm0 = np.full(nseg, np.nan)
            Tpx = np.full(nseg, np.nan)
            Tm01 = np.full(nseg, np.nan)
            Tm02 = np.full(nseg, np.nan)
            Flag = np.zeros(nseg, dtype=bool)
            fgrid_list = []
            SB_list = []

            for i in range(nseg):
                idx = slice(i * samplesInHour, (i + 1) * samplesInHour)
                seg = IG_ts[idx]

                # down-cross stats
                H, Tp_est = _zdc_segment_metrics(seg, Fs)
                if np.isfinite(H): Hm0[i] = H
                if np.isfinite(Tp_est): Tpx[i] = Tp_est
                s1 = seg[:-1]; s2 = seg[1:]
                crossings = np.where((s1 > 0) & (s2 <= 0))[0]
                Flag[i] = (len(crossings) - 1) > 50

                # spectral stats on filtered IG segment
                f, S = _welch_psd(seg - np.nanmean(seg), Fs)
                Hm0_w, Tm01_w, Tm02_w, Tp_w = _band_moments(f, S, (1/250.0, 1/25.0))
                fgrid_list.append(f)
                SB_list.append(S)
                if np.isfinite(Hm0_w): Hm0[i] = Hm0_w
                if np.isfinite(Tm01_w): Tm01[i] = Tm01_w
                if np.isfinite(Tm02_w): Tm02[i] = Tm02_w
                if np.isfinite(Tp_w):   Tpx[i]  = Tp_w

            IG.zdc["Pressure"].Hm0 = Hm0
            IG.zdc["Pressure"].Tm01 = Tm01
            IG.zdc["Pressure"].Tm02 = Tm02
            IG.zdc["Pressure"].Tp = Tpx
            IG.zdc["Pressure"].Flag_hour = Flag
            if fgrid_list:
                fgrid = fgrid_list[0]
                SB = np.vstack([np.interp(fgrid, fgrid_list[i], SB_list[i]) for i in range(nseg)])
                IG.zdc["Pressure"].f = fgrid
                IG.zdc["Pressure"].SB = SB

        return IG, Pressure, AST

    # ----------------- wavelet analysis -----------------
    def _run_wavelet(IG: IGGroup,
                     pressure_raw: Optional[np.ndarray],
                     ast_raw: Optional[np.ndarray]) -> IGGroup:

        def cwt_accumulate(x: np.ndarray,
                           fs: float,
                           freq_limits: Tuple[float, float],
                           voices_per_octave: int,
                           seg_sec: int,
                           hop_sec: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            N = len(x)
            Nwin = int(round(seg_sec * fs))
            Nhop = int(round(hop_sec * fs))
            if N < Nwin:
                # single window fallback
                Nwin = N
                Nhop = N

            dt = 1.0 / fs
            fmin, fmax = freq_limits
            # Build a log-spaced frequency grid aligned with voices_per_octave
            num_octaves = np.log2(fmax / fmin)
            n_voices_total = max(int(np.ceil(num_octaves * voices_per_octave)) + 1, 1)
            freqs = fmin * (2 ** (np.arange(n_voices_total) / float(voices_per_octave)))
            freqs = freqs[(freqs >= fmin) & (freqs <= fmax)]
            # Convert desired frequencies to scales for Morlet
            fc = pywt.central_frequency('morl')
            scales = fc / (freqs * dt)

            F = len(freqs)
            power_accum = np.zeros((F, N), dtype=np.float32)
            cover = np.zeros(N, dtype=np.int32)

            start = 0
            while start + Nwin <= N:
                idx = slice(start, start + Nwin)
                seg = x[idx]
                # PyWavelets returns coef matrix of shape (len(scales), len(seg))
                coef, _ = pywt.cwt(seg, scales, 'morl', sampling_period=dt)
                P = (np.abs(coef) ** 2).astype(np.float32)
                power_accum[:, idx] += P
                cover[idx] += 1
                start += Nhop

            # normalize by coverage
            cover[cover == 0] = 1
            power = power_accum / cover[None, :]

            t = np.arange(N) * dt
            varx = np.nanvar(x)
            pow_frac = (power / varx) if varx > 0 else np.full_like(power, np.nan)
            pow_dB = 10.0 * np.log10(power + np.finfo(float).eps)

            return freqs, t, power, pow_frac, pow_dB

        # AST
        if ast_raw is not None and len(ast_raw):
            f, t, pow_, pow_frac, pow_dB = cwt_accumulate(
                ast_raw, Fs, wavelet_freq_limits, wavelet_voices_per_octave,
                wavelet_seg_sec, wavelet_hop_sec
            )
            IG.Wavelet["AST"] = WaveletResults(f=f, t=t, pow=pow_, pow_frac=pow_frac, pow_dB=pow_dB)

        # Pressure
        if pressure_raw is not None and len(pressure_raw):
            f, t, pow_, pow_frac, pow_dB = cwt_accumulate(
                pressure_raw, Fs, wavelet_freq_limits, wavelet_voices_per_octave,
                wavelet_seg_sec, wavelet_hop_sec
            )
            IG.Wavelet["Pressure"] = WaveletResults(f=f, t=t, pow=pow_, pow_frac=pow_frac, pow_dB=pow_dB)

        return IG

    # ===================== Run per data availability =====================
    IG, Pressure, AST = _run_spectral(pressure, ast)
    IG, Pressure, AST = _run_zdc(IG, Pressure, AST, pressure, ast)
    IG = _run_wavelet(IG, pressure, ast)

    return IG, Pressure, AST


# ----------------- (optional) quick usage example -----------------
if __name__ == "__main__":
    # Example with fake data
    Fs = 2.0  # Hz
    T = 3 * 3600  # 3 hours
    t = np.arange(int(Fs * T)) / Fs

    # Fake AST: IG component (0.01 Hz, 100 s period) + noise
    ast = 0.2 * np.sin(2 * np.pi * 0.01 * t) + 0.02 * np.random.randn(t.size)

    # Fake Pressure: hydrostatic equivalent to ~0.2 m surface oscillation
    rho, g = 1025.0, 9.81
    pressure = (ast * rho * g) + 50.0  # add DC offset

    IG, P, A = process_igw_nortek(t, Fs, pressure=pressure, ast=ast)
    print("AST IG Hm0 (hourly):", IG.Spectral["AST"]["Hm0"] if "AST" in IG.Spectral else None)
    print("Pressure IG Hm0 (hourly):", IG.Spectral["Pressure"]["Hm0"] if "Pressure" in IG.Spectral else None)
