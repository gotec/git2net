#%%
import pyce
import pathpy as pp
import pandas as pd

import os

from datetime import datetime

#%%
if os.path.exists('out.db'):
  os.remove('out.db')

pyce.mine_git_repo('.', 'out.db')


#%% COEDITING NETWORK
t = pyce.get_coediting_network('out.db')
print(t)
n = pp.Network.from_temporal_network(t)
print(n)
pp.visualisation.export_html(n, 'coediting_net.html', width=600, height=800)


#%% 
t = pyce.get_coediting_network('out.db')
print(t)

t = pyce.get_coediting_network('out.db', time_to=datetime.strptime('2019-01-14 14:50:34', '%Y-%m-%d %H:%M:%S'))
print(t)


#%%
n = pyce.get_bipartite_network('out.db')
print(n)
colors = {}
for x in n.nodes:
    if ' ' in x:
        colors[x] = 'red'
    else:
        colors[x] = 'green'
pp.visualisation.export_html(n, 'bipartite.html', width=600, height=800, node_color=colors)


#%% DAG
dag = pyce.get_dag('out.db')
colors = {}
for x in dag.nodes:
    colors[x] = 'lightblue'
for x in dag.roots:
    colors[x] = 'red'
for x in dag.leafs:
    colors[x] = 'green'
n = pp.Network()
for e in dag.edges:
    n.add_edge(e[0], e[1])
pp.visualisation.export_html(dag, 'dag.html', width=600, height=800, node_color=colors)

#%%
node_mapping = {}
for n in dag.nodes:
    node_mapping[n] = n.split(',')[0]

p = pp.path_extraction.paths_from_dag(dag, node_mapping)
print(p)

mog = pp.MultiOrderModel(p, max_order=3)
mog.estimate_order(p)

#%%
