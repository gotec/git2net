#%%
import os
import importlib
import git2net
importlib.reload(git2net)

db_path = 'out.db'

if os.path.exists(db_path):
    os.remove(db_path)

git2net.mine_git_repo('.', db_path, num_processes=8, chunksize=1, use_blocks=False)












#%%
import pydriller
git_repo = pydriller.GitRepository('.')
commit = git_repo.get_commit('7de40e6c54bae5f0f43bd056b710eac3e2153fc7')
for modification in commit.modifications:
    print(modification)

#%%
import pandas as pd

d = {}
len(d.keys()) == 0

#%%
