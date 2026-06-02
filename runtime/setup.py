"""Build native extensions (Phase 15).

Metadata and package discovery live in pyproject.toml; this file exists so
``python3 setup.py build_ext --inplace`` works without an editable install.
"""

from setuptools import setup

setup()
