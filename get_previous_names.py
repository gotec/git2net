#%%
import pandas as pd
import sqlite3
import pathpy as pp

con = sqlite3.connect('out.db')

df = pd.read_sql("SELECT mod_filename, mod_old_path, mod_new_path FROM edits", con)

dag = pp.DAG()

df.drop_duplicates(inplace=True)
df = df.loc[df.mod_old_path != df.mod_new_path, :]

for _, edit in df.drop_duplicates().iterrows():
    #if ()
    dag.add_edge(edit.mod_old_path, edit.mod_new_path)

dag


#%%
