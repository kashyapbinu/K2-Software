"""
K2 Aerospace — Dynamics Analysis Package
==========================================
Flutter, vibration, and aeroelastic analysis.
"""
from dynamics.flutter_analysis import flutter_speed, flutter_analysis, FlutterResult
from dynamics.vibration_analysis import (frequency_response, miles_equation,
                                          VibrationResult, random_vibration_response)
from dynamics.aeroelastic import (divergence_speed, aeroelastic_effectiveness,
                                   AeroelasticResult, full_aeroelastic_analysis)

__all__ = [
    "flutter_speed", "flutter_analysis", "FlutterResult",
    "frequency_response", "miles_equation", "VibrationResult",
    "random_vibration_response",
    "divergence_speed", "aeroelastic_effectiveness", "AeroelasticResult",
    "full_aeroelastic_analysis",
]
