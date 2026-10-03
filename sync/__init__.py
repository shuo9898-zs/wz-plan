"""CARLA-SUMO synchronization layer."""
from .bridge import CarlaSumoCoordinateBridge
from .ego_proxy import EgoProxySynchronizer
# Pedestrian proxy is retained as an archived implementation but is not part of
# the active sync API.  S4 controls its jaywalker entirely inside CARLA.
# from .pedestrian_proxy import PedestrianProxySynchronizer
from .background_traffic import BackgroundTrafficSynchronizer

__all__ = [
    "CarlaSumoCoordinateBridge",
    "EgoProxySynchronizer",
    # "PedestrianProxySynchronizer",
    "BackgroundTrafficSynchronizer",
]
