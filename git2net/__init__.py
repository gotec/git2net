"""
An OpenSource Python package for the extraction of fine-grained
and time-stamped co-editing networks from git repositories.
"""

__author__ = "Christoph Gote"
__email__ = "cgote@ethz.ch"
__version__ = "1.0.0"

from .git2net import get_commit_dag
from .git2net import identify_file_renaming
from .git2net import mine_git_repo
from .git2net import extract_editing_paths
from .git2net import extract_edits
from .git2net import process_commit
from .git2net import identify_edits
from .git2net import extract_edits_merge
from .git2net import get_edited_file_paths_since_split
