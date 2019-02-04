#%%
import pathpy as pp
import importlib
import git2net
importlib.reload(git2net)


sqlite_db_file = 'out.db'

dag, paths, node_info, edge_info = git2net.extract_editing_paths(sqlite_db_file, with_start=False, filenames=['.gitignore'])

pp.visualisation.plot(dag, width=1000, height=1000, node_color=node_info['colors'])



#%%
git2net.extract_editing_paths

#%%
print(paths.summary())

#%%
