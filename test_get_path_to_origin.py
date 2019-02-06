#%%
import importlib
import git2net
importlib.reload(git2net)
import pydriller
import pathpy as pp

post_line_num = 42
file_path = 'manuscript.tex'
commit_hash = 'a368f245b4c2bb125dfcdf8593a29c4e157feeee'


git_repo = pydriller.GitRepository('../kdd-anomalies')
commit = git_repo.get_commit(commit_hash)
mod = commit.modifications[0]


blame = git_repo.git.blame(commit_hash + '^', '--', file_path).split('\n')

blame_fields = blame[post_line_num - 1].split(' ')
original_commit_hash = blame_fields[0].replace('^', '')


git2net.get_path_to_origin(git_repo, original_commit_hash, commit_hash)


#%%
