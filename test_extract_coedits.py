#%%
import importlib
import git2net
importlib.reload(git2net)
import pydriller

repo_string = '.'
commit_hash = 'd3a26c0062b116bc2280a7728025665932f3c959'
filename = 'pyce.py'

git_repo = pydriller.GitRepository(repo_string)
commit = git_repo.get_commit(commit_hash)
for mod in commit.modifications:
    if mod.filename == filename:
        df = git2net.extract_coedits(git_repo, commit, mod, use_blocks=True)
df

#%%
