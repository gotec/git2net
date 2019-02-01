#%%
import git2net

git2net.mine_git_repo('.', 'out.db', no_parallel=True)

#%%
import os
import importlib
importlib.reload(git2net)
import git2net

db_path = 'out.db'

# if os.path.exists(db_path):
#     os.remove(db_path)

git2net.mine_git_repo('../igraph', db_path, num_processes=8, chunksize=1)

#%%
with open('out.db') as f:
    print(f)


#%%

#%%
