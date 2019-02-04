#%%
import importlib
import git2net
importlib.reload(git2net)
import pydriller

repo_string = '.'
commit_hash = 'bc0a43ab286643b5ea43dcb9b8434c51a2ee427d'
filename = '.gitignore'

git_repo = pydriller.GitRepository(repo_string)
commit = git_repo.get_commit(commit_hash)
for mod in commit.modifications:
    if mod.filename == filename:
        df = git2net.extract_edits(git_repo, commit, mod, use_blocks=False)
df

#%%
