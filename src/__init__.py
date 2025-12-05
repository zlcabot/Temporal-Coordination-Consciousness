"""
Temporal Coordination Capacity - Consciousness Studies

Core modules for computing temporal coordination measures and
analyzing coordination across consciousness states.

Implements the Two-Stage Model of Conscious Access.
"""

from .coordination_measures import (
    compute_mutual_information,
    compute_transfer_entropy,
    compute_phi_d,
    compute_phi_f,
    compute_phi_c
)

from .cross_channel_analysis import (
    compute_cross_channel_coordination,
    compute_plv,
    compute_cross_channel_mi,
    two_stage_model_analysis
)

from .statistical_analysis import (
    compute_cohens_d,
    compute_effect_size_ci,
    subject_level_aggregation
)

__version__ = '1.0.0'
__author__ = 'Zayin Cabot'
