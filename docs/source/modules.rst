=======
Modules
=======

git2net provides four modules—`extraction`, `disambiguation`, `visualisation`, and `complexity`.


----------
Extraction
----------

The module `extraction` contains all the functions that operate directly on a git repository.
The most important functions in this module are:

    * **mine_git_repo**: mines edits from a locally cloned git repositry to an SQLite database.
    * **mine_github**: creates a local clone and mines edits from repository on GitHub to an SQLite database.
    * **check_mining_complete**: checks if a repository has been fully mined.
    * **mining_state_summary**: provides information on any commits that have not been fully mined.

Checkout the `API reference <https://git2net.readthedocs.io/en/latest/api_reference.html#module-git2net.extraction>`_ for information on the complete list of available functions.


--------------
Disambiguation
--------------

The module `disambiguation` only contains a single function which allows you to disambiguate author identities.
The disambiguation is based on the algorithm `gambit`_.

.. _gambit: https://github.com/gotec/gambit


    * **disambiguate_aliases_db**: disambiguates author aliases in a database mined with git2net.


-------------
Visualisation
-------------

The `visualisation` module provides functions to generate various network projections based on the SQLite database created during the mining process.

    * **get_coediting_network**: creates a co-editing network where nodes are authors who are connected by a directed link if they consecutively edited the same line of code. Links are directed from the previous to the subsequent author.
    * **get_coauthorship_network**: creates a co-authorship network were nodes are authors who are connected by an undirected link if the edited the same file.
    * **get_bipartite_network**: creates a bipartite network with nodes representing authors and files. Undirected links exist between an author and all files the author edited.
    * **get_line_editing_paths**: creates paths for all lines in a repository. The paths contain ordered sequences of authors who subsequently edited a line. The number of paths generated for a line depends on the number of forks and merges the line was involved in.
    * **get_commit_editing_dag**: creates a directed acyclic graph where nodes are commits. Commits are connected by a directed link if a one commit modifies lines last editied in another commit. Links are directed from the editing commit to the edited commit.


----------
Complexity
----------

The module `complexity` provides the functionality to compute a variety of complexity measures for the commits and files in a git repository. Specifically, for all commits, we compute the number of editing events (`events`) and the total Levenshtein edit distance (`levenshtein_distance`) for all modified files. In addition, we compute the Halstead effort (`HE`), the cyclomatic complexity (`CCN`), the number of lines of code (`NLOC`), the number of tokens (`TOK`), and the number of functions (`FUN`) in all modified files before (`*_pre`) and after (`*_post`) each commit. We further compute the change (`*_delta`) for all complexity measures. As we show in `this publication <https://arxiv.org/abs/2201.04588>`_, the absolute value of the change in complexity can be used as a proxy for the productivity of developers in Open Source software projects.


    * **compute_complexity**: computes complexity measures for all mined commit/file combinations in a database mined with git2net.