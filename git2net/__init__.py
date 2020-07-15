"""
An OpenSource Python package for the extraction of fine-grained
and time-stamped co-editing networks from git repositories.
"""

__author__ = "Christoph Gote"
__email__ = "cgote@ethz.ch"
__version__ = "1.3.2"

from .extraction import mine_git_repo
from .extraction import get_unified_changes
from .extraction import get_commit_dag
from .extraction import identify_file_renaming
from .extraction import text_entropy
from .extraction import mining_state_summary
from .extraction import check_mining_complete
from .visualisation import get_line_editing_paths
from .visualisation import get_commit_editing_paths
from .visualisation import get_coediting_network
from .visualisation import get_coauthorship_network
from .visualisation import get_bipartite_network
