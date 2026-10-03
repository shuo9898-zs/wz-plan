"""Portable Aug-24 Encoder-1 + PPO + held-out validation experiment.

Use ``run_train.py`` or ``run_dashboard.py`` as the standalone entry points.
They activate the frozen top-level runtime before importing experiment code.
Keeping package import side-effect free also lets repository tests inspect this
package without globally replacing their normal ``baseline``/``env`` modules.
"""
