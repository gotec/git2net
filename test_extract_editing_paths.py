#%%
import pathpy as pp
import importlib
import git2net
importlib.reload(git2net)
import numpy as np


# sqlite_db_file = 'kdd_anomalies.db'

# dag, paths, node_info, edge_info = git2net.extract_editing_paths(sqlite_db_file, with_start=True,
#                                     filenames=['manuscript.tex', 'method.tex', 'data.tex'])

sqlite_db_file = 'out.db'

dag, paths, node_info, edge_info = git2net.extract_editing_paths(sqlite_db_file, with_start=False,
                                    filenames=['test_git2net.py'])

pp.visualisation.export_html(dag, 'out.html', width=1500, height=1500, node_color=node_info['colors'])

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
file_info_count = {}
for filename in df.filenames.unique():
    file_info_count[filename] = {}
    for author in df.authors.unique():
        file_info_count[filename][author] = {}
        file_info_count[filename][author]['initial'] = len(df.loc[(df.authors == author) &
                                                            (df.colors == 'initial') &
                                                            (df.filenames == filename), :])
        file_info_count[filename][author]['intermediate'] = len(df.loc[(df.authors == author) &
                                                                 (df.colors == 'intermediate') &
                                                                 (df.filenames == filename), :])
        file_info_count[filename][author]['final'] = len(df.loc[(df.authors == author) &
                                                          (df.colors == 'final') &
                                                          (df.filenames == filename), :])
        file_info_count[filename][author]['deleted'] = len(df.loc[(df.authors == author) &
                                                            (df.colors == 'deleted') &
                                                            (df.filenames == filename), :])
        file_info_count[filename][author]['initial and final'] = len(df.loc[(df.authors == author) &
                                                                (df.colors == 'initial and final') &
                                                                (df.filenames == filename), :])


file_info_dist = {}
for filename in df.filenames.unique():
    file_info_dist[filename] = {}
    for author in df.authors.unique():
        file_info_dist[filename][author] = {}
        file_info_dist[filename][author]['initial'] = np.nansum(df.loc[(df.authors == author) &
                                                            (df.colors == 'initial') &
                                                            (df.filenames == filename), :].edit_distance)
        file_info_dist[filename][author]['intermediate'] = np.nansum(df.loc[(df.authors == author) &
                                                                 (df.colors == 'intermediate') &
                                                                 (df.filenames == filename), :].edit_distance)
        file_info_dist[filename][author]['final'] = np.nansum(df.loc[(df.authors == author) &
                                                          (df.colors == 'final') &
                                                          (df.filenames == filename), :].edit_distance)
        file_info_dist[filename][author]['deleted'] = np.nansum(df.loc[(df.authors == author) &
                                                            (df.colors == 'deleted') &
                                                            (df.filenames == filename), :].edit_distance)
        file_info_dist[filename][author]['initial and final'] = np.nansum(df.loc[(df.authors == author) &
                                                                (df.colors == 'initial and final') &
                                                                (df.filenames == filename), :].edit_distance)

#%%
for filename in file_info_count.keys():
    bottom = [0,0,0,0,0]
    for author in file_info_count[filename].keys():
        plt.bar(file_info_count[filename][author].keys(), file_info_count[filename][author].values(),
                bottom=bottom)
        bottom = [sum(x) for x in zip(bottom, file_info_count[filename][author].values())]
    plt.legend(file_info_count[filename].keys())
    plt.title(filename)
    plt.show()


#%%
for filename in file_info_dist.keys():
    bottom = [0,0,0,0,0]
    for author in file_info_dist[filename].keys():
        plt.bar(file_info_dist[filename][author].keys(), file_info_dist[filename][author].values(),
                bottom=bottom)
        bottom = [sum(x) for x in zip(bottom, file_info_dist[filename][author].values())]
    plt.legend(file_info_dist[filename].keys())
    plt.title(filename)
    plt.show()


#%%


#%%
