#%%
import pathpy as pp
import importlib
import git2net
importlib.reload(git2net)
import numpy as np


# sqlite_db_file = 'kdd_anomalies.db'

# dag, paths, node_info, edge_info = git2net.extract_editing_paths(sqlite_db_file, with_start=True,
#                                     file_paths=['manuscript.tex', 'method.tex', 'data.tex'])

sqlite_db_file = 'out.db'

# dag, paths, node_info, edge_info = git2net.extract_editing_paths(sqlite_db_file, with_start=True, file_paths=['.gitignore', 'ignore'])
dag, paths, node_info, edge_info = git2net.extract_editing_paths(sqlite_db_file, with_start=True)

# pp.visualisation.plot(dag, width=1500, height=1500, node_color=node_info['colors'])

#%%
print(paths.summary())

#%%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(node_info)
df = df.loc[df.colors != 'gray', :]
df = df.loc[pd.isnull(df.authors) == False, :]
df.replace('#218380', 'initial', inplace=True)
df.replace('#73D2DE', 'intermediate', inplace=True)
df.replace('#2E5EAA', 'final', inplace=True)
df.replace('#A8322D', 'deleted', inplace=True)
df.replace('#5B4E77', 'initial and final', inplace=True)
df.replace('giona', 'Giona Casiraghi', inplace=True)
df.replace('fschweitzer-ETHZ', 'Frank Schweitzer', inplace=True)
df.replace('eliassi', 'Tina Eliassi-Rad', inplace=True)

#%%
import sqlite3
import importlib
import git2net
importlib.reload(git2net)
sqlite_db_file = 'out.db'
con = sqlite3.connect(sqlite_db_file)
path = con.execute("SELECT repository FROM _metadata").fetchall()[0][0]
dag, aliases = git2net.identify_file_renaming(path)
dag


#%%
file_info_count = {}
for file_path in df.file_paths.unique():
    file_info_count[file_path] = {}
    for author in df.authors.unique():
        file_info_count[file_path][author] = {}
        file_info_count[file_path][author]['initial'] = len(df.loc[(df.authors == author) &
                                                            (df.colors == 'initial') &
                                                            (df.file_paths == file_path), :])
        file_info_count[file_path][author]['intermediate'] = len(df.loc[(df.authors == author) &
                                                                 (df.colors == 'intermediate') &
                                                                 (df.file_paths == file_path), :])
        file_info_count[file_path][author]['final'] = len(df.loc[(df.authors == author) &
                                                          (df.colors == 'final') &
                                                          (df.file_paths == file_path), :])
        file_info_count[file_path][author]['deleted'] = len(df.loc[(df.authors == author) &
                                                            (df.colors == 'deleted') &
                                                            (df.file_paths == file_path), :])
        file_info_count[file_path][author]['initial and final'] = len(df.loc[(df.authors == author) &
                                                                (df.colors == 'initial and final') &
                                                                (df.file_paths == file_path), :])


file_info_dist = {}
for file_path in df.file_paths.unique():
    file_info_dist[file_path] = {}
    for author in df.authors.unique():
        file_info_dist[file_path][author] = {}
        file_info_dist[file_path][author]['initial'] = np.nansum(df.loc[(df.authors == author) &
                                                            (df.colors == 'initial') &
                                                            (df.file_paths == file_path), :].edit_distance)
        file_info_dist[file_path][author]['intermediate'] = np.nansum(df.loc[(df.authors == author) &
                                                                 (df.colors == 'intermediate') &
                                                                 (df.file_paths == file_path), :].edit_distance)
        file_info_dist[file_path][author]['final'] = np.nansum(df.loc[(df.authors == author) &
                                                          (df.colors == 'final') &
                                                          (df.file_paths == file_path), :].edit_distance)
        file_info_dist[file_path][author]['deleted'] = np.nansum(df.loc[(df.authors == author) &
                                                            (df.colors == 'deleted') &
                                                            (df.file_paths == file_path), :].edit_distance)
        file_info_dist[file_path][author]['initial and final'] = np.nansum(df.loc[(df.authors == author) &
                                                                (df.colors == 'initial and final') &
                                                                (df.file_paths == file_path), :].edit_distance)

#%%
for file_path in file_info_count.keys():
    bottom = [0,0,0,0,0]
    for author in file_info_count[file_path].keys():
        plt.bar(file_info_count[file_path][author].keys(), file_info_count[file_path][author].values(),
                bottom=bottom)
        bottom = [sum(x) for x in zip(bottom, file_info_count[file_path][author].values())]
    plt.legend(file_info_count[file_path].keys())
    plt.title(file_path)
    plt.show()


#%%
for file_path in file_info_dist.keys():
    bottom = [0,0,0,0,0]
    for author in file_info_dist[file_path].keys():
        plt.bar(file_info_dist[file_path][author].keys(), file_info_dist[file_path][author].values(),
                bottom=bottom)
        bottom = [sum(x) for x in zip(bottom, file_info_dist[file_path][author].values())]
    plt.legend(file_info_dist[file_path].keys())
    plt.title(file_path)
    plt.show()

####################################################################################################

#%%
import pydriller
import importlib
import git2net
importlib.reload(git2net)
path = '../kdd-anomalies'
dag, aliases = git2net.identify_file_renaming(path)
dag

#%%
aliases




#%%
