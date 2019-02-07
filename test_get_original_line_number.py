#%%
import importlib
import git2net
importlib.reload(git2net)
import pydriller

post_line_num = 102
file_name = 'test.py'
commit_hash = 'db0e20b413363cb93447ed4567bb8e959fc7f306'


git_repo = pydriller.GitRepository('.')
commit = git_repo.get_commit(commit_hash)
mod = commit.modifications[0]


blame = git_repo.git.blame(commit_hash + '^', '--', 'test.py').split('\n')

blame_fields = blame[post_line_num - 1].split(' ')
original_commit_hash = blame_fields[0].replace('^', '')
print(original_commit_hash)

aliases = {'test.py': ['git2net.py', 'pyce.py', 'test.py']}

git2net.get_original_line_number(git_repo, file_name, original_commit_hash, commit_hash + '^', post_line_num, aliases)


#%%
