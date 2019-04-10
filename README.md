# git2net

`git2net` is an Open Source Python package that facilitates the extraction of co-editing networks
from git repositories.

## Download and installation

`git2net` is pure `python` code. It has no platform-specific dependencies and thus works on all
platforms. Assuming you are using `pip`, you can install latest version of `git2net` by running:

```
> pip install git2net
```

This also installs the necessary dependencies.

## How to use git2net

We have publised some motivating results as well as details on the mining algorithm in "git2net - Mining Time-Stamped Co-Editing Networks from Large git Repositories. Together with the paper we have released a jupyter notebook reproducing the majority of the results shown in the paper on [zenodo.org](https://zenodo.org/record/2587483#.XK4LPENoSCg). This notebook also serves as tutorial introducing the functionality of `git2net`.

## How to cite git2net

```
@article{DBLP:journals/corr/abs-1903-10180,
  author    = {Christoph Gote and
               Ingo Scholtes and
               Frank Schweitzer},
  title     = {git2net - Mining Time-Stamped Co-Editing Networks from Large git Repositories},
  journal   = {CoRR},
  volume    = {abs/1903.10180},
  year      = {2019},
  url       = {http://arxiv.org/abs/1903.10180},
  archivePrefix = {arXiv},
  eprint    = {1903.10180},
  timestamp = {Mon, 01 Apr 2019 14:07:37 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1903-10180},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## License

This software is licensed under the GNU Affero General Public License v3 (AGPL-3.0).
