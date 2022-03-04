===============
Getting Started
===============

A central aim of git2net is allowing you to conveniently obtain and visualise network projections of editing activity in git repositories.
Let's have a look at an example on how you can achieve this:

.. sourcecode:: python

    import git2net
    import pathpy as pp
    
    github_url = 'gotec/git2net'
    git_repo_dir = 'git2net4analysis'
    sqlite_db_file = 'git2net4analysis.db'
    
    # Clone and mine repository from GitHub
    git2net.mine_github(github_url, git_repo_dir, sqlite_db_file)
    
    # Disambiguate author aliases in the resulting database
    git2net.disambiguate_aliases_db(sqlite_db_file)
    
    # Obtain temporal bipartite network representing authors editing files over time
    t, node_info, edge_info = git2net.get_bipartite_network(sqlite_db_file, time_from=time_from)
    
    # Aggregate to a static network
    n = pp.Network.from_temporal_network(t)
    
    # Visualise the resulting network
    colour_map = {'author': '#73D2DE', 'file': '#2E5EAA'}
    node_color = {node: colour_map[node_info['class'][node]] for node in n.nodes}
    pp.visualisation.plot(n, node_color=node_color)
    
In the example above, we used three functions of git2net.
First, we extract edits from the repository using `mine_github`.
Then, we disambiguate author identities using `disambiguate_aliases_db`.
Finally, we visualise the bipartite author-file network with `get_bipartite_network`.

Corresponding to the calls above, git2net's functionality is partitionied into three modules: `extraction`, `disambiguation`, `visualisation`, and `complexity`.
We outline the most important functions of each module `here <https://git2net.readthedocs.io/en/latest/modules.html>`_.
For a comprehensive details on all functions of git2net we refer to the `API reference <https://git2net.readthedocs.io/en/latest/api_reference.html>`_.


---------
Tutorials
---------

To help you get started, we provide an extensive set of tutorials covering different aspects of analysing your repository with `git2net`.
You can directly interact with the notebooks in *Binder*, or view them in *NBViewer* via the links below.

* `Open all tutorials in Binder <https://mybinder.org/v2/gh/gotec/git2net-tutorials/HEAD>`_
* `Open all tutorials in NBViewer <https://nbviewer.org/github/gotec/git2net-tutorials/tree/main/>`_

In addition, we provide links to the individual tutorial notebooks in the tabs below:

.. tabs::

   .. tab:: Cloning
      
      We show how to clone and prepare a git repository for analysis with git2net.
      
      * `Open the cloning tutorial in Binder <https://mybinder.org/v2/gh/gotec/git2net-tutorials/HEAD?labpath=1_Cloning_Git_Repositories.ipynb>`_
      * `Open the cloning tutorial in Google Colab <https://colab.research.google.com/github/gotec/git2net-tutorials/blob/master/1_Cloning_Git_Repositories.ipynb>`_
      * `Open the cloning tutorial in NBViewer <https://nbviewer.org/github/gotec/git2net-tutorials/tree/main/1_Cloning_Git_Repositories.ipynb>`_
        
   .. tab:: Mining
   
      We introduce the mining options git2net provides and discuss the resulting SQLite database.
      
      * `Open the mining tutoral in Binder <https://mybinder.org/v2/gh/gotec/git2net-tutorials/HEAD?labpath=2_Mining_Git_Repositories.ipynb>`_
      * `Open the mining tutoral in Google Colab <https://colab.research.google.com/github/gotec/git2net-tutorials/blob/master/2_Mining_Git_Repositories.ipynb>`_
      * `Open the mining tutoral in NBViewer <https://nbviewer.org/github/gotec/git2net-tutorials/tree/main/2_Mining_Git_Repositories.ipynb>`_
   
   .. tab:: Disambiguation
   
      We explain how you can use git2net to disambiguate author aliases in a mined repository using `gambit <https://github.com/gotec/gambit>`_.
      
      * `Open the dismabiguation tutorial in Binder <https://mybinder.org/v2/gh/gotec/git2net-tutorials/HEAD?labpath=3_Author_Disambiguation.ipynb>`_
      * `Open the dismabiguation tutorial in Google Colab <https://colab.research.google.com/github/gotec/git2net-tutorials/blob/master/3_Author_Disambiguation.ipynb>`_
      * `Open the dismabiguation tutorial in NBViewer <https://nbviewer.org/github/gotec/git2net-tutorials/tree/main/3_Author_Disambiguation.ipynb>`_
        
   .. tab:: Networks
   
      We show how you can use git2net to generate various network projects of the edits in a mined repository.
      
      * `Open the network analysis tutorial in Binder <https://mybinder.org/v2/gh/gotec/git2net-tutorials/HEAD?labpath=4_Network_Analysis.ipynb>`_
      * `Open the network analysis tutorial in Google Colab <https://colab.research.google.com/github/gotec/git2net-tutorials/blob/master/4_Network_Analysis.ipynb>`_
      * `Open the network analysis tutorial in NBViewer <https://nbviewer.org/github/gotec/git2net-tutorials/tree/main/4_Network_Analysis.ipynb>`_
        
   .. tab:: Database
   
      We show how you can the database mined by git2net to answer elaborate questions on the activity in git repositories.
      
      * `Open the database analysis tutorial in Binder <https://mybinder.org/v2/gh/gotec/git2net-tutorials/HEAD?labpath=5_Database_Analysis.ipynb>`_
      * `Open the database analysis tutorial in Google Colab <https://colab.research.google.com/github/gotec/git2net-tutorials/blob/master/5_Database_Analysis.ipynb>`_
      * `Open the database analysis tutorial in NBViewer <https://nbviewer.org/github/gotec/git2net-tutorials/tree/main/5_Database_Analysis.ipynb>`_


--------------
Usage Examples
--------------

We have published some motivating results as well as details on the mining algorithm in `"git2net - Mining Time-Stamped Co-Editing Networks from Large git Repositories" <https://dl.acm.org/doi/10.1109/MSR.2019.00070>`_.

In `"Analysing Time-Stamped Co-Editing Networks in Software Development Teams using git2net" <https://link.springer.com/article/10.1007/s10664-020-09928-2>`_, we use `git2net` to mine more than 1.2 million commits of over 25,000 developers. We use this data to test a hypothesis on the relation between developer productivity and co-editing patterns in software teams.

Finally, in `"Big Data = Big Insights? Operationalising Brooks' Law in a Massive GitHub Data Set" <https://arxiv.org/abs/2201.04588>`_, we mine a corpus containing over 200 GitHub repositories using `git2net`. Based on the resulting data, we study the relationship between team size and productivity in OSS development teams. If you want to use this extensive data set for your own study, we made it publicly available on `zenodo.org <https://doi.org/10.5281/zenodo.5294965>`_.
