#%%
import importlib
import git2net
importlib.reload(git2net)
import pydriller

repo_string = 'test_repos/test_repo_1'
commit_hash = 'dcf060d5aa93077c84552ce6ed56a0f0a37e4dca'
filename = 'text_file.txt'

git_repo = pydriller.GitRepository(repo_string)
commit = git_repo.get_commit(commit_hash)
for mod in commit.modifications:
    if mod.filename == filename:
        df = git2net.extract_edits(git_repo, commit, mod, use_blocks=False)
df

#%%
df['original_commit_deletion']

#%%
repo_string = 'test_repos/test_repo_1'
git_repo = pydriller.GitRepository(repo_string)
commit = git_repo.get_commit('dcf060d5aa93077c84552ce6ed56a0f0a37e4dca')
edited_file_path = 'text_file.txt'

for modification in commit.modifications:
    modification_info = {}
    modification_info['filename'] = modification.filename
    modification_info['new_path'] = modification.new_path
    modification_info['old_path'] = modification.old_path
    modification_info['cyclomatic_complexity_of_file'] = modification.complexity
    modification_info['lines_of_code_in_file'] = modification.nloc
    modification_info['modification_type'] = modification.change_type.name

    edits_info = git2net.extract_edits_merge(git_repo, commit, modification_info, use_blocks=False, blame_C='-C')

edits_info

#%%
