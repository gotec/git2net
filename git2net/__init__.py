"""
An OpenSource Python package for the extraction of fine-grained
and time-stamped co-editing networks from git repositories.
"""

import importlib.metadata

__author__ = "Christoph Gote"
__email__ = "cgote@ethz.ch"
__version__ = importlib.metadata.version('git2net')

from .extraction import mine_git_repo  # noqa
from .extraction import mine_github  # noqa
from .extraction import get_unified_changes  # noqa
from .extraction import get_commit_dag  # noqa
from .extraction import identify_file_renaming  # noqa
from .extraction import text_entropy  # noqa
from .extraction import mining_state_summary  # noqa
from .extraction import check_mining_complete  # noqa
from .disambiguation import disambiguate_aliases_db  # noqa
from .visualisation import get_line_editing_paths  # noqa
from .visualisation import get_commit_editing_dag  # noqa
from .visualisation import get_coediting_network  # noqa
from .visualisation import get_coauthorship_network  # noqa
from .visualisation import get_bipartite_network  # noqa
from .complexity import compute_complexity  # noqa

import logging

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s]  %(name)s:%(levelname)-10s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
