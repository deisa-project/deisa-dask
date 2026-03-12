###################################################################################################
# Copyright (c) 2026 Commissariat a l'énergie atomique et aux énergies alternatives (CEA)
# SPDX-License-Identifier: MIT
###################################################################################################
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("deisa-dask")  # installed version
except PackageNotFoundError:
    from .__version__ import __version__  # fallback

from .bridge import Bridge
from .deisa import Deisa
from .utils import get_connection_info
