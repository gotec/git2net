#%%
import importlib
import git2net
importlib.reload(git2net)
import pydriller

post_line_num = 89
file_name = 'manuscript.tex'
commit_hash = '782b228d005a2c8accf32a4b46a32ec2fcef3caa'


git_repo = pydriller.GitRepository('../kdd-anomalies')
commit = git_repo.get_commit(commit_hash)
for mod in commit.modifications:
    if mod.filename == file_name:
        break

assert len(commit.parents) == 1

blame = git_repo.git.blame(commit.parents[0], '--', file_name).split('\n')
blame_fields = blame[post_line_num - 1].split(' ')
original_commit_hash = blame_fields[0].replace('^', '')
print(original_commit_hash)

aliases = {'manuscript.tex': ['manuscript.tex']}

git2net.get_original_line_number(git_repo, file_name, original_commit_hash, commit_hash, post_line_num, aliases)

#%%

#%%
import pydriller
file_name = 'test_file.txt'
commit_hash = '6bbb37b5'

git_repo = pydriller.GitRepository('../test_how_git_works')
commit = git_repo.get_commit(commit_hash)
for mod in commit.modifications:
    if mod.filename == file_name:
        break

blame = git_repo.git.blame(commit.hash, ['-C', '--show-number', '--porcelain'], file_name)
blame

#%%
import pydriller
import git2net
importlib.reload(git2net)

file_name = 'manuscript.tex'
commit_hash = '3fc047e34d2be73b169bc77ebda28ca7ddf47b19'

git_repo = pydriller.GitRepository('../kdd-anomalies')
commit = git_repo.get_commit(commit_hash)
for mod in commit.modifications:
    if mod.filename == file_name:
        break

blame = git_repo.git.blame(commit.hash, ['-C', '--show-number', '--porcelain'], file_name)
blame_info = git2net.parse_porcelain_blame(blame)
blame_info

#%%
import pandas as pd
for i in pd.to_numeric(blame_info['post line']):
    print(blame_info.loc[pd.to_numeric(blame_info['post line']) == i, 'original line'])


#%%
pd.to_numeric(blame_info['post line'])==1

#%%
type(blame_info['post line'][0])

#%%
