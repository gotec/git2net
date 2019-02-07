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


dag, colors = git2net.get_paths_to_origin(git_repo, original_commit_hash, commit_hash)

pp.visualisation.plot(dag, width=1000, height=1000, node_color=colors)

#%%
aliases = {'manuscript.tex': ['manuscript.tex']}
git2net.get_original_line_number(git_repo, 'manuscript.tex', original_commit_hash, commit_hash + '^', 42, aliases)


#%%
paths = [['a','b'],['c','d','e','e']]
[set(x) for x in l]

set(node for path in paths for node in path)

#%%
import pydriller
git_repo = pydriller.GitRepository('../test_how_git_works')
blame = git_repo.git.blame('d527074b^', 'test_file.txt').split('\n')
blame

#%%
git_repo.parse_diff(git_repo.get_commit('d527074b').modifications[0].diff)

# git_repo.get_commit('e467d9a').parents


#%%
