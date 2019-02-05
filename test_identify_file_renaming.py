#%%
import importlib
import git2net
importlib.reload(git2net)

dag, aliases = git2net.identify_file_renaming('.')
dag

#%%
aliases


#%%
