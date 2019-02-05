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
