#%%
import importlib
import git2net
importlib.reload(git2net)

added_lines = {1: '', 2: '', 7: ''}
deleted_lines = {1: '', 4: ''}

pre_to_post, edits = git2net.identify_edits(deleted_lines, added_lines, use_blocks=True)

#%%
pre_to_post

#%%
edits.head()

#%%
