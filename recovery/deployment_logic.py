"""
K2 AeroSim — Deployment Logic
"""

class DeploymentLogic:
    """
    Handles when to trigger drogue and main deployments.
    """
    def __init__(self, main_deploy_altitude: float = 300.0, drogue_delay: float = 0.0):
        self.main_deploy_altitude = main_deploy_altitude
        self.drogue_delay = drogue_delay
        
    def should_deploy_drogue(self, phase: str, time_since_apogee: float) -> bool:
        """Evaluate if drogue should deploy based on apogee + delay."""
        # This is a stub for the integration with flight_phases
        return phase == "APOGEE" and time_since_apogee >= self.drogue_delay

    def should_deploy_main(self, phase: str, altitude: float, velocity_z: float) -> bool:
        """Evaluate if main should deploy based on altitude and descent rate."""
        # Only deploy main if we are descending and below the threshold
        return phase == "DROGUE_DESCENT" and velocity_z < 0 and altitude <= self.main_deploy_altitude
