"""Compatibility alias for scaffold-style environment imports."""

from server.pollution_exposure_minimizer_environment import (
    PollutionExposureMinimizerEnvironment,
)

MyEnvironment = PollutionExposureMinimizerEnvironment

__all__ = ["PollutionExposureMinimizerEnvironment", "MyEnvironment"]
