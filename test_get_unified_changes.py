#%%
import importlib
import git2net
importlib.reload(git2net)

git2net.get_unified_changes('.', 'c657e752b411caf531e3fff8fc0ea8e0b756ed43', 'git2net.py')


#%%
import git2net
if git2net.get_unified_changes(a, b, c):
    print(1)

#%%
