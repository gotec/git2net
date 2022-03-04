[![Tests](https://github.com/gotec/git2net/actions/workflows/python-app.yml/badge.svg)](https://github.com/gotec/git2net/actions/workflows/python-app.yml)
[![Documentation Status](https://readthedocs.org/projects/git2net/badge/?version=latest)](https://git2net.readthedocs.io/en/latest/?badge=latest)


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

### Tutorials

After installation, we suggest to check out our [tutorials](https://github.com/gotec/git2net-tutorials), detailing how to get started using `git2net`.
We provide tutorials covering different aspects of analysing your repository with `git2net`.
You can directly interact with the notebooks in *Binder*, or view them in *NBViewer* via the badges below.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gotec/git2net-tutorials/HEAD)
[![NBViewer](https://img.shields.io/badge/View%20on-nbviewer-informational)](https://nbviewer.org/github/gotec/git2net-tutorials/tree/main/)

In addition, we provide links to the individual tutorial notebooks below:

| Tutorial | Binder | Google Colab | NBViewer |
| :---     | :---:    | :---:  | :---: |
| 1. Cloning a repository for analysis | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gotec/git2net-tutorials/HEAD?labpath=1_Cloning_Git_Repositories.ipynb) | [![Open Cloning Tutorial In Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gotec/git2net-tutorials/blob/master/1_Cloning_Git_Repositories.ipynb) | [![NBViewer](https://img.shields.io/badge/View%20on-nbviewer-informational)](https://nbviewer.org/github/gotec/git2net-tutorials/tree/main/1_Cloning_Git_Repositories.ipynb) |
| 2. Mining git repositories with [`git2net`](https://github.com/gotec/git2net) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gotec/git2net-tutorials/HEAD?labpath=2_Mining_Git_Repositories.ipynb) | [![Open Mining Tutorial In Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gotec/git2net-tutorials/blob/master/2_Mining_Git_Repositories.ipynb) | [![NBViewer](https://img.shields.io/badge/View%20on-nbviewer-informational)](https://nbviewer.org/github/gotec/git2net-tutorials/tree/main/2_Mining_Git_Repositories.ipynb) |
| 3. Author disambiguation with [`gambit`](https://github.com/gotec/gambit) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gotec/git2net-tutorials/HEAD?labpath=3_Author_Disambiguation.ipynb) | [![Open Disambiguation Tutorial In Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gotec/git2net-tutorials/blob/master/3_Author_Disambiguation.ipynb) | [![NBViewer](https://img.shields.io/badge/View%20on-nbviewer-informational)](https://nbviewer.org/github/gotec/git2net-tutorials/tree/main/3_Author_Disambiguation.ipynb) |
| 4. Network analysis with [`pathpy`](https://www.pathpy.net/) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gotec/git2net-tutorials/HEAD?labpath=4_Network_Analysis.ipynb) | [![Open Network Tutorial In Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gotec/git2net-tutorials/blob/master/4_Network_Analysis.ipynb) | [![NBViewer](https://img.shields.io/badge/View%20on-nbviewer-informational)](https://nbviewer.org/github/gotec/git2net-tutorials/tree/main/4_Network_Analysis.ipynb) |
| 5. Database-based analyses | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gotec/git2net-tutorials/HEAD?labpath=5_Database_Analysis.ipynb) | [![Open Database Tutorial In Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gotec/git2net-tutorials/blob/master/5_Database_Analysis.ipynb) | [![NBViewer](https://img.shields.io/badge/View%20on-nbviewer-informational)](https://nbviewer.org/github/gotec/git2net-tutorials/tree/main/5_Database_Analysis.ipynb) |
| 6. Computing file complexity [`git2net`](https://github.com/gotec/git2net) | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/gotec/git2net-tutorials/HEAD?labpath=6_Computing_Complexities.ipynb) | [![Open Database Tutorial In Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gotec/git2net-tutorials/blob/master/6_Computing_Complexities.ipynb) | [![NBViewer](https://img.shields.io/badge/View%20on-nbviewer-informational)](https://nbviewer.org/github/gotec/git2net-tutorials/tree/main/6_Computing_Complexities.ipynb) |

### Documentation

`git2net`'s documentation is available at [git2net.readthedocs.io](https://git2net.readthedocs.io).

### Usage examples

We have published some motivating results as well as details on the mining algorithm in ["git2net - Mining Time-Stamped Co-Editing Networks from Large git Repositories"](https://dl.acm.org/doi/10.1109/MSR.2019.00070).

In ["Analysing Time-Stamped Co-Editing Networks in Software Development Teams using git2net"](https://link.springer.com/article/10.1007/s10664-020-09928-2), we use `git2net` to mine more than 1.2 million commits of over 25,000 developers. We use this data to test a hypothesis on the relation between developer productivity and co-editing patterns in software teams.

Finally, in ["Big Data = Big Insights? Operationalising Brooks' Law in a Massive GitHub Data Set"](https://arxiv.org/abs/2201.04588), we mine a corpus containing over 200 GitHub repositories using `git2net`. Based on the resulting data, we study the relationship between team size and productivity in OSS development teams. If you want to use this extensive data set for your own study, we made it publicly available on [zenodo.org](https://doi.org/10.5281/zenodo.5294965).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5294965.svg)](https://doi.org/10.5281/zenodo.5294965)

## How to cite git2net

```
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
```


## License

This software is licensed under the GNU Affero General Public License v3 (AGPL-3.0).
