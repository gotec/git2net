=========================
Overview and Installation
=========================

git2net is an Open Source Python package that facilitates the extraction of co-editing networks from git repositories.


------------
Requirements
------------

git2net is pure `Python`_ code. It has no platform-specific dependencies and thus works on all
platforms. The only requirement is a version of `Git >= 2.0`.

.. _Python: https://www.python.org
.. _Git: https://git-scm.com/


------------------
Installing git2net
------------------

Assuming you are using `pip`_, you can install latest version of `git2net` from the command-line by running:


.. _pip: https://pip.pypa.io/en/latest/installing.html

.. sourcecode:: none

    $ pip install git2net 

This command also installs the necessary dependencies. Among other dependencies, which are listed as `install_requires` in `git2net's setup file <https://github.com/gotec/git2net/blob/main/setup.cfg>`_, git2net depends on the `python-Levenshtein <https://github.com/ztane/python-Levenshtein>`_ package to compute Levenshtein distances for edited lines of code.
On sytems running Windows, automatically compiling this C based module might fail during installation.
In this case, unofficial Windows binaries can be found `here <https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-levenshtein>`_, which might help you get started.


-----------------------
Contributing to git2net
-----------------------

The source code for git2net is available in a repository on GitHub which can be browsed at:

* https://github.com/gotec/git2net

If you find any bugs related to git2net please report them as issues there.

git2net is developed as an Open Source project.
This means that your ideas and inputs are highly welcome.
Feel free to share the project and contribute yourself.
To get started, you can clone git2net's repository as follows:

.. sourcecode:: none

    $ git clone git@github.com:gotec/git2net.git

Now uninstall any existing version of git2net and install a local version based on the cloned repository:

.. sourcecode:: none

    $ pip uninstall git2net
    $ cd git2net
    $ pip install -e .
    
This will also install git2net's dependencies.

git2net provides a set of tests that you should run before creating a pull request.
To do so, you will first need to unzip the test resitory they are based on:

.. sourcecode:: none

    $ unzip test_repos/test_repo_1.zip -d test_repos/

Then, you can run the tests with:

.. sourcecode:: none

    $ pytest


--------------
Citing git2net
--------------

.. sourcecode:: none

    @inproceedings{gote2019git2net,
        title={git2net: {M}ining time-stamped co-editing networks from large git repositories},
        author={Gote, Christoph and Scholtes, Ingo and Schweitzer, Frank},
        booktitle={Proceedings of the 16th International Conference on Mining Software Repositories},
        pages={433--444},
        year={2019},
        organization={IEEE Press}
    }

    @article{gote2021analysing,
        title={Analysing time-stamped co-editing networks in software development teams using git2net},
        author={Gote, Christoph and Scholtes, Ingo and Schweitzer, Frank},
        journal={Empirical Software Engineering},
        volume={26},
        number={4},
        pages={1--41},
        year={2021},
        publisher={Springer}
    }
    
    
-------
License
-------

This software is licensed under the GNU Affero General Public License v3 (AGPL-3.0).