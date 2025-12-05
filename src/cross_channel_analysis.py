"""
Cross-Channel Analysis Module for Consciousness Studies

Functions for computing hemispheric coordination and analyzing
coordination capacity across consciousness states.

This module implements the cross-channel operationalization of
temporal coordination measures for EEG analysis.

References:
    Cabot, Z. (2025). Temporal Coordination Capacity Reveals 
    Integration Without Reportability. Neuroscience of Consciousness.
"""

import numpy as np
from scipy.signal import hilbert
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Optional
from .coordination_measures import (
    compute_mutual_information,
    compute_entropy,
    compute_phi_c
)


def compute_plv(signal1: np.ndarray, signal2: np.ndarray, 
                fs: float = 256, 
                freq_range: Tuple[float, float] = (4, 30)) -> float:
    """
    Compute Phase-Locking Value between two signals.
    
    PLV measures the consistency of phase difference between signals,
    used as the cross-channel operationalization of Φ_f.
    
    Parameters
    ----------
    signal1 : np.ndarray
        First EEG channel
    signal2 : np.ndarray
        Second EEG channel
    fs : float
        Sampling frequency (default: 256 Hz)
    freq_range : tuple
        Frequency range for filtering (default: 4-30 Hz)
    
    Returns
    -------
    float
        PLV value in [0, 1]
    """
    from scipy.signal import butter, filtfilt
    
    # Bandpass filter
    nyq = fs / 2
    low = freq_range[0] / nyq
    high = freq_range[1] / nyq
    b, a = butter(4, [low, high], btype='band')
    
    filtered1 = filtfilt(b, a, signal1)
    filtered2 = filtfilt(b, a, signal2)
    
    # Extract instantaneous phase via Hilbert transform
    phase1 = np.angle(hilbert(filtered1))
    phase2 = np.angle(hilbert(filtered2))
    
    # PLV = |mean(exp(i * phase_diff))|
    phase_diff = phase1 - phase2
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
    
    return plv


def compute_cross_channel_mi(signal1: np.ndarray, 
                              signal2: np.ndarray,
                              k: int = 4) -> float:
    """
    Compute cross-channel mutual information.
    
    Used as the operationalization of cross-channel Φ_d.
    
    Parameters
    ----------
    signal1 : np.ndarray
        First EEG channel
    signal2 : np.ndarray
        Second EEG channel
    k : int
        Number of neighbors for k-NN estimation
    
    Returns
    -------
    float
        Normalized mutual information in [0, 1]
    """
    s1 = signal1.reshape(-1, 1)
    s2 = signal2.reshape(-1, 1)
    
    mi = compute_mutual_information(s1, s2, k=k)
    h1 = compute_entropy(s1, k=k)
    h2 = compute_entropy(s2, k=k)
    
    # Normalize by geometric mean of entropies
    norm = np.sqrt(h1 * h2)
    if norm > 0:
        nmi = mi / norm
    else:
        nmi = 0
    
    return np.clip(nmi, 0, 1)


def compute_cross_channel_coordination(
    eeg_data: Dict[str, np.ndarray],
    channel_pairs: List[Tuple[str, str]] = None,
    fs: float = 256,
    k: int = 4
) -> Dict:
    """
    Compute cross-channel coordination measures.
    
    Parameters
    ----------
    eeg_data : dict
        Dictionary mapping channel names to time series
    channel_pairs : list, optional
        List of channel pairs to analyze
    fs : float
        Sampling frequency
    k : int
        k-NN parameter
    
    Returns
    -------
    dict
        Coordination measures for each pair and aggregates
    """
    if channel_pairs is None:
        # Default: bilateral pairs
        channel_pairs = [
            ('C3', 'C4'),
            ('F3', 'F4'),
            ('O1', 'O2')
        ]
    
    results = {'pairs': {}}
    phi_d_values = []
    phi_f_values = []
    phi_c_values = []
    
    for ch1, ch2 in channel_pairs:
        if ch1 not in eeg_data or ch2 not in eeg_data:
            continue
        
        s1 = eeg_data[ch1]
        s2 = eeg_data[ch2]
        
        # Cross-channel Φ_d via MI
        phi_d = compute_cross_channel_mi(s1, s2, k=k)
        
        # Cross-channel Φ_f via PLV
        phi_f = compute_plv(s1, s2, fs=fs)
        
        # Coordination capacity
        phi_c = compute_phi_c(phi_d, phi_f)
        
        results['pairs'][f'{ch1}-{ch2}'] = {
            'phi_d': phi_d,
            'phi_f': phi_f,
            'phi_c': phi_c
        }
        
        phi_d_values.append(phi_d)
        phi_f_values.append(phi_f)
        phi_c_values.append(phi_c)
    
    # Aggregates
    results['mean_phi_d'] = np.mean(phi_d_values) if phi_d_values else np.nan
    results['mean_phi_f'] = np.mean(phi_f_values) if phi_f_values else np.nan
    results['mean_phi_c'] = np.mean(phi_c_values) if phi_c_values else np.nan
    
    # Coordination balance ratio
    total = results['mean_phi_d'] + results['mean_phi_f'] + results['mean_phi_c']
    results['R'] = results['mean_phi_c'] / total if total > 0 else np.nan
    
    # Temporal orientation
    results['tau'] = (results['mean_phi_f'] / results['mean_phi_d'] 
                      if results['mean_phi_d'] > 0 else np.nan)
    
    return results


def analyze_sleep_stages(
    eeg_epochs: Dict[str, Dict[str, np.ndarray]],
    fs: float = 256
) -> Dict:
    """
    Analyze coordination across sleep stages.
    
    Parameters
    ----------
    eeg_epochs : dict
        Nested dict: stage -> epoch_id -> channel -> data
    fs : float
        Sampling frequency
    
    Returns
    -------
    dict
        Stage-level coordination statistics
    """
    stage_results = {}
    
    for stage, epochs in eeg_epochs.items():
        phi_c_values = []
        phi_d_values = []
        phi_f_values = []
        R_values = []
        
        for epoch_id, channels in epochs.items():
            result = compute_cross_channel_coordination(channels, fs=fs)
            
            if not np.isnan(result['mean_phi_c']):
                phi_c_values.append(result['mean_phi_c'])
                phi_d_values.append(result['mean_phi_d'])
                phi_f_values.append(result['mean_phi_f'])
                R_values.append(result['R'])
        
        stage_results[stage] = {
            'n': len(phi_c_values),
            'phi_c_mean': np.mean(phi_c_values),
            'phi_c_sd': np.std(phi_c_values, ddof=1),
            'phi_d_mean': np.mean(phi_d_values),
            'phi_f_mean': np.mean(phi_f_values),
            'R_mean': np.mean(R_values),
            'R_sd': np.std(R_values, ddof=1)
        }
    
    return stage_results


def compute_temporal_orientation(phi_d: float, phi_f: float) -> Tuple[float, str]:
    """
    Compute temporal orientation (τ) and interpret.
    
    τ = Φ_f / Φ_d
    
    Parameters
    ----------
    phi_d : float
        Duration measure
    phi_f : float
        Frequency measure
    
    Returns
    -------
    tuple
        (τ value, interpretation string)
    """
    if phi_d <= 0:
        return np.inf, "Frequency-only (infinite τ)"
    
    tau = phi_f / phi_d
    
    if tau < 0.8:
        interpretation = "Duration-dominant (past-constrained)"
    elif tau < 1.2:
        interpretation = "Balanced (temporal equilibrium)"
    else:
        interpretation = "Frequency-dominant (future-oriented)"
    
    return tau, interpretation


def two_stage_model_analysis(phi_c: float, 
                              access_available: bool) -> Dict:
    """
    Apply two-stage model of consciousness.
    
    Stage 1: Conscious moment ⟺ Φ_c(t) > θ_event
    Stage 2: Reportable ⟺ Stage 1 AND access available
    
    Parameters
    ----------
    phi_c : float
        Coordination capacity value
    access_available : bool
        Whether access function is available
    
    Returns
    -------
    dict
        Two-stage model assessment
    """
    theta_event = 0.5  # Coordination threshold
    
    stage1_met = phi_c > theta_event
    reportable = stage1_met and access_available
    
    return {
        'phi_c': phi_c,
        'theta_event': theta_event,
        'stage1_conscious_moment': stage1_met,
        'access_available': access_available,
        'reportable': reportable,
        'interpretation': _interpret_two_stage(stage1_met, access_available)
    }


def _interpret_two_stage(stage1: bool, access: bool) -> str:
    """Generate interpretation string for two-stage model."""
    if not stage1:
        return "Below coordination threshold - no conscious moment"
    elif stage1 and access:
        return "Conscious moment WITH access - reportable experience"
    elif stage1 and not access:
        return "Conscious moment WITHOUT access - integration without reportability"
    else:
        return "Unknown state"


# Example usage
if __name__ == '__main__':
    import numpy as np
    
    # Generate synthetic EEG data
    np.random.seed(42)
    fs = 256
    duration = 30  # seconds
    t = np.linspace(0, duration, fs * duration)
    
    # Simulate bilateral EEG with some common source
    common = 10 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
    
    eeg_data = {
        'C3': common + 5 * np.random.randn(len(t)),
        'C4': common + 5 * np.random.randn(len(t)),
        'F3': common * 0.8 + 5 * np.random.randn(len(t)),
        'F4': common * 0.8 + 5 * np.random.randn(len(t)),
    }
    
    result = compute_cross_channel_coordination(eeg_data, fs=fs)
    
    print("Cross-Channel Coordination Analysis")
    print("=" * 40)
    print(f"Mean Φ_d: {result['mean_phi_d']:.3f}")
    print(f"Mean Φ_f: {result['mean_phi_f']:.3f}")
    print(f"Mean Φ_c: {result['mean_phi_c']:.3f}")
    print(f"R (balance): {result['R']:.3f}")
    print(f"τ (orientation): {result['tau']:.3f}")
    
    # Two-stage model
    tsm = two_stage_model_analysis(result['mean_phi_c'], access_available=True)
    print(f"\nTwo-Stage Model: {tsm['interpretation']}")
