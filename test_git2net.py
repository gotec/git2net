#%%
import os
import importlib
import git2net
importlib.reload(git2net)

db_path = 'kdd_anomalies.db'

# if os.path.exists(db_path):
#     os.remove(db_path)

# git2net.mine_git_repo('.', db_path, use_blocks=False, extract_original_line_num=True)
git2net.mine_git_repo('../kdd-anomalies', db_path, use_blocks=False, extract_original_line_num=True)










#%%
import pydriller
git_repo = pydriller.GitRepository('../kdd-anomalies')
commit = git_repo.get_commit('a368f245b4c2bb125dfcdf8593a29c4e157feeee')
for i in range(10):
    commit = git_repo.get_commit(commit.hash + '^')
    print(commit.hash)

#%%
import pandas as pd

pd.DataFrame().empty




#%%
