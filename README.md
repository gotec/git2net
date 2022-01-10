[![Open Tutorial In Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gotec/git2net/blob/master/TUTORIAL.ipynb)
[![View tutorial on nbviewer](https://img.shields.io/badge/View%20on-nbviewer-informational)](https://nbviewer.org/github/gotec/git2net/blob/master/TUTORIAL.ipynb)

# git2net

`git2net` is an Open Source Python package that facilitates the extraction of co-editing networks
from git repositories.

## Download and installation

`git2net` is pure `python` code. It has no platform-specific dependencies and thus works on all
platforms. The only requirement is a version of `git >= 2.0`. Assuming you are using `pip`, you can install latest version of `git2net` by running:

```
> pip install git2net
```

This also installs the necessary dependencies. `git2net` depends on the `python-Levenshtein` package to compute Levenshtein distances for edited lines of code. On sytems running Windows, automatically compiling this C based module might fail during installation. In this case, unofficial Windows binaries can be found [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#python-levenshtein), which might help you get started.

## How to use git2net
After installation, we suggest to check out our [tutorial](https://github.com/gotec/git2net/blob/master/TUTORIAL.ipynb), detailing how to get started using `git2net`. We also provide detailed inline documentation serving as reference.

In addition, we have publised some motivating results as well as details on the mining algorithm in ["git2net - Mining Time-Stamped Co-Editing Networks from Large git Repositories"](https://dl.acm.org/doi/10.1109/MSR.2019.00070). Together with the paper, we have further released a jupyter notebook (using an early version of `git2net`) reproducing the majority of the results shown in the paper on [zenodo.org](https://zenodo.org/record/2587483#.XK4LPENoSCg).

All functions of `git2net`have been tested on Ubuntu, Mac OS, and Windows.

## How to cite git2net

```
@inproceedings{gote2019git2net,
  title={git2net: mining time-stamped co-editing networks from large git repositories},
  author={Gote, Christoph and Scholtes, Ingo and Schweitzer, Frank},
  booktitle={Proceedings of the 16th International Conference on Mining Software Repositories},
  pages={433--444},
  year={2019},
  organization={IEEE Press}
}
```

## License

This software is licensed under the GNU Affero General Public License v3 (AGPL-3.0).
