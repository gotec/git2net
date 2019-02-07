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
