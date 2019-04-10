#!/usr/bin/python3

#####################################################
# All functions that work on the extracted database #
#####################################################

import pydriller
import pandas as pd
import git2net
import pathpy as pp
import sqlite3
import datetime

def get_line_editing_paths(sqlite_db_file, commit_hashes=None, file_paths=False, with_start=False,
                           merge_renaming=True):
    """ Returns line editing DAG as well as line editing paths.

        Node and edge infos set up to be expanded with future releases.

    Args:
        sqlite_db_file: path to sqlite database mined with git2net line method
        commit_hashes: list of commits to consider, by default all commits are considered
        file_paths: list of files to consider, by defailt all files are considered
        with_start: bool, determines if node for filename is included as start for all editing pahts
        merge_renaming: bool, determines if file renaming is considered

    Returns:
        paths: line editing pahts, pathpy Path object
        dag: line editing directed acyclic graph, pathpy DAG object
        node_info: info on node charactaristics
        edge_info: info on edge characteristics
    """

    # Connect to provided database.
    con = sqlite3.connect(sqlite_db_file)

    # Check if database is valid.
    try:
        path = con.execute("SELECT repository FROM _metadata").fetchall()[0][0]
        method = con.execute("SELECT method FROM _metadata").fetchall()[0][0]
        if method == 'blocks':
            raise Exception("Invalid database. A database mined with 'use_blocks=False' is " +
                            "required.")
    except sqlite3.OperationalError:
        raise Exception("You either provided no database or a database not created with git2net. " +
                        "Please provide a valid datatabase mined with 'use_blocks=False'.")

    # Extract required data from the provided database.
    commits = pd.read_sql("""SELECT hash, author_name, author_date FROM commits""", con)
    edits = pd.read_sql("""SELECT levenshtein_dist,
                                  old_path,
                                  new_path,
                                  commit_hash,
                                  original_commit_deletion,
                                  original_commit_addition,
                                  original_line_no_deletion,
                                  original_line_no_addition,
                                  original_file_path_deletion,
                                  original_file_path_addition,
                                  post_starting_line_no,
                                  edit_type
                           FROM edits""", con)

    # Filter edits table if only edits from specific commits are considered.
    if commit_hashes:
        edits = edits.loc[[x in commit_hashes for x in edits.commit_hash], :]

    # Rename file paths to latest name if option is selected.
    if merge_renaming:
        # Identify files that have been renamed.
        _, aliases = git2net.identify_file_renaming(path)
        # Update their name in the edits table.
        for key, value in aliases.items():
            edits.replace(key, value[0], inplace=True)

    # Filter edits table if specific files are considered. Has to be done after renaming.
    if not file_paths == False:
        edits = edits.loc[[x in file_paths for x in edits.new_path], :]

    dag = pp.DAG()
    node_info = {}
    node_info['colors'] = {}
    # node_info['authors'] = {}
    # node_info['file_paths'] = {}
    # node_info['edit_distance'] = {}
    edge_info = {}
    edge_info['colors'] = {}
    # edge_info['weights'] = {}

    # Get author and date of deletions.
    edits = pd.merge(edits, commits, how='left', left_on='original_commit_deletion',
                            right_on='hash').drop(['hash'], axis=1)
    edits.rename(columns = {'author_name':'author_name_deletion',
                            'author_date': 'author_date_deletion'}, inplace = True)

    # Get author and date of additions.
    edits = pd.merge(edits, commits, how='left', left_on='original_commit_addition',
                            right_on='hash').drop(['hash'], axis=1)
    edits.rename(columns = {'author_name':'author_name_addition',
                            'author_date': 'author_date_addition'}, inplace = True)

     # Get current author and date
    edits = pd.merge(edits, commits, how='left', left_on='commit_hash',
                            right_on='hash').drop(['hash'], axis=1)

    # Sort edits by author date.
    edits.sort_values('author_date', ascending=True, inplace=True)

    file_paths = set()

    for _, edit in edits.iterrows():
        if edit.edit_type == 'replacement':
            # Generate name of target node.
            target = 'L' + str(int(edit.post_starting_line_no)) + ' ' + \
                     edit.new_path + ' ' + \
                     edit.commit_hash[0:7]

            # Source of deletion must exist.
            source_deletion = 'L' + str(int(edit.original_line_no_deletion)) + ' ' + \
                              edit.original_file_path_deletion + ' ' + \
                              edit.original_commit_deletion[0:7]
            dag.add_edge(source_deletion, target)
            edge_info['colors'][(source_deletion, target)] = 'white'
            # Check id source of addition exists.
            if edit.original_commit_addition is not None:
                source_addition = 'L' + str(int(edit.original_line_no_addition)) + ' ' + \
                                  edit.original_file_path_addition + ' ' + \
                                  edit.original_commit_addition[0:7]
                dag.add_edge(source_addition, target)
                edge_info['colors'][(source_addition, target)] = '#FBB13C' # yellow
        elif edit.edit_type == 'deletion':
            # An edit in a file can only change lines in that file, not in the file the line was
            # copied from.
            if edit.original_file_path_deletion == edit.old_path:
                # Generate name of target node.
                target = 'deleted L' + str(int(edit.original_line_no_deletion)) + ' ' + \
                        edit.original_file_path_deletion + ' ' + \
                        edit.original_commit_deletion[0:7]

                # Source of deletion must exist.
                source_deletion = 'L' + str(int(edit.original_line_no_deletion)) + ' ' + \
                                edit.original_file_path_deletion + ' ' + \
                                edit.original_commit_deletion[0:7]
                dag.add_edge(source_deletion, target)
                edge_info['colors'][(source_deletion, target)] = 'white'
            else:
                copied_from = 'L' + str(int(edit.original_line_no_deletion)) + ' ' + \
                                edit.original_file_path_deletion + ' ' + \
                                edit.original_commit_deletion[0:7]
                found_copied_to = False
                for copied_to in dag.successors[copied_from]:
                    if copied_to.split(' ')[1] == edit.old_path:
                        found_copied_to = True
                        break
                assert found_copied_to
                dag.add_edge(copied_to, 'deleted ' + copied_to)
                edge_info['colors'][(copied_to, 'deleted ' + copied_to)] = 'white'
        elif edit.edit_type == 'addition':
            # Generate name of target node.
            target = 'L' + str(int(edit.post_starting_line_no)) + ' ' + \
                     edit.new_path + ' ' + \
                     edit.commit_hash[0:7]

            # Add file path as source and add file path to file_paths list.
            source = edit.new_path
            file_paths.add(source)
            dag.add_edge(source, target)
            edge_info['colors'][(source, target)] = 'gray'

            # Check id source of addition exists.
            if edit.original_commit_addition is not None:
                source_addition = 'L' + str(int(edit.original_line_no_addition)) + ' ' + \
                                  edit.original_file_path_addition + ' ' + \
                                  edit.original_commit_addition[0:7]
                dag.add_edge(source_addition, target)
                edge_info['colors'][(source_addition, target)] = '#FBB13C'
        elif edit.edit_type == 'file_renaming':
            pass
        else:
            raise Exception("Unexpected error in 'extract_editing_paths'.")

    for node in dag.nodes:
        if node in file_paths:
            node_info['colors'][node] = 'gray'
        elif '#FBB13C' in [edge_info['colors'][n] for n in [(x, node)
                                                  for x in dag.predecessors[node]]]:
            node_info['colors'][node] = '#FBB13C' # yellow
        elif node.startswith('deleted'):
            node_info['colors'][node] = '#A8322D' # red
        elif 'white' not in [edge_info['colors'][n] for n in [(node, x)
                                                    for x in dag.successors[node]]]:
            node_info['colors'][node] = '#2E5EAA' # blue
        elif not dag.predecessors[node].isdisjoint(file_paths):
            node_info['colors'][node] = '#218380' # green
        else:
            node_info['colors'][node] = '#73D2DE' # light blue

    if not with_start:
        for file_path in file_paths:
            dag.remove_node(file_path)

    dag.topsort()

    assert dag.is_acyclic is True

    paths = pp.path_extraction.paths_from_dag(dag)

    return paths, dag, node_info, edge_info


def get_commit_editing_paths(sqlite_db_file, time_from=None, time_to=None, filename=None):
    """ Returns DAG of commits where an edge between commit A and B indicates that lines written in
        commit A where changes in commit B. Further outputs editing paths extracted from the DAG.

        Node and edge infos set up to be expanded with future releases.

    Args:
        sqlite_db_file: path to sqlite database
        time_from: start time of time window filter, datetime object
        time_to: end time of time window filter, datetime object
        filename: filter to obtain only commits editing a certain file

    Returns:
        paths: pathpy path object capturing editing paths
        dag: pathpy dag object linking commits
        node_info: info on node charactaristics
        edge_info: info on edge characteristics
    """

    con = sqlite3.connect(sqlite_db_file)
    data = pd.read_sql("""SELECT edits.original_commit_deletion AS pre_commit,
                                 edits.commit_hash AS post_commit,
                                 edits.filename,
                                 commits.author_date AS time
                          FROM edits
                          JOIN commits
                          ON edits.commit_hash = commits.hash""", con).drop_duplicates()
    if filename is not None:
        data = data.loc[data.filename==filename, :]

    if time_from == None:
        time_from = datetime.datetime.strptime(min(data.time), '%Y-%m-%d %H:%M:%S')
    if time_to == None:
        time_to = datetime.datetime.strptime(max(data.time), '%Y-%m-%d %H:%M:%S')

    node_info = {}
    edge_info = {}

    dag = pp.DAG()
    for idx, row in data.iterrows():
        if (datetime.datetime.strptime(row.time, '%Y-%m-%d %H:%M:%S') >= time_from) and \
           (datetime.datetime.strptime(row.time, '%Y-%m-%d %H:%M:%S') <= time_to):
            dag.add_edge(row.pre_commit, row.post_commit)

    dag.topsort()

    assert dag.is_acyclic is True

    paths = pp.path_extraction.paths_from_dag(dag)

    return paths, dag, node_info, edge_info


def get_coediting_network(db_location, time_from=None, time_to=None):
    """ Returns coediting network containing links between authors who coedited at least one line of
        code within a given time window.

        Node and edge infos set up to be expanded with future releases.

    Args:
        sqlite_db_file: path to sqlite database
        time_from: start time of time window filter, datetime object
        time_to: end time of time window filter, datetime object

    Returns:
        t: pathpy temporal network
        node_info: info on node charactaristics
        edge_info: info on edge characteristics
    """

    con = sqlite3.connect(db_location)
    edits = pd.read_sql("""SELECT original_commit_deletion AS pre_commit,
                                  commit_hash AS post_commit,
                                  levenshtein_dist
                           FROM edits""", con).drop_duplicates()
    commits = pd.read_sql("""SELECT hash, author_name, author_date FROM commits""", con)

    data = pd.merge(edits, commits, how='left', left_on='pre_commit', right_on='hash') \
                    .drop(['pre_commit', 'hash', 'author_date'], axis=1)
    data.columns = ['post_commit', 'levenshtein_dist', 'pre_author']
    data = pd.merge(data, commits, how='left', left_on='post_commit', right_on='hash') \
                    .drop(['post_commit', 'hash'], axis=1)
    data.columns = ['levenshtein_dist', 'pre_author', 'post_author', 'time']
    data = data[['pre_author', 'post_author', 'time', 'levenshtein_dist']]

    if time_from == None:
        time_from = datetime.datetime.strptime(min(data.time), '%Y-%m-%d %H:%M:%S')
    if time_to == None:
        time_to = datetime.datetime.strptime(max(data.time), '%Y-%m-%d %H:%M:%S')

    node_info = {}
    edge_info = {}

    t = pp.TemporalNetwork()
    for idx, row in data.iterrows():
        if (datetime.datetime.strptime(row.time, '%Y-%m-%d %H:%M:%S') >= time_from) and \
           (datetime.datetime.strptime(row.time, '%Y-%m-%d %H:%M:%S') <= time_to) and not \
           (row['post_author'] == row['pre_author']):
            t.add_edge(row['post_author'],
                       row['pre_author'],
                       row['time'],
                       directed=True,
                       timestamp_format='%Y-%m-%d %H:%M:%S')

    return t, node_info, edge_info


def get_coauthorship_network(sqlite_db_file, time_from=None, time_to=None):
    """ Returns coauthorship network containing links between authors who coedited at least one code
        file within a given time window.

        Node and edge infos set up to be expanded with future releases.

    Args:
        sqlite_db_file: path to sqlite database
        time_from: start time of time window filter, datetime object
        time_to: end time of time window filter, datetime object

    Returns:
        n: pathpy network
        node_info: info on node charactaristics
        edge_info: info on edge characteristics
    """

    con = sqlite3.connect(sqlite_db_file)
    edits = pd.read_sql("""SELECT original_commit_deletion AS pre_commit,
                                  commit_hash AS post_commit,
                                  filename
                           FROM edits""", con)
    commits = pd.read_sql("""SELECT hash, author_name, author_date AS time FROM commits""", con)

    data_pre = pd.merge(edits, commits, how='left', left_on='pre_commit', right_on='hash') \
                    .drop(['pre_commit', 'post_commit', 'hash'], axis=1)
    data_post = pd.merge(edits, commits, how='left', left_on='post_commit', right_on='hash') \
                    .drop(['pre_commit', 'post_commit', 'hash'], axis=1)
    data = pd.concat([data_pre, data_post])

    if time_from == None:
        time_from = datetime-datetime.strptime(min(data['time']), '%Y-%m-%d %H:%M:%S')
    if time_to == None:
        time_to = datetime-datetime.strptime(max(data['time']), '%Y-%m-%d %H:%M:%S')

    data = data.loc[pd.to_datetime(data['time']) >= time_from, :]
    data = data.loc[pd.to_datetime(data['time']) <= time_to, :]

    node_info = {}
    edge_info = {}

    n = pp.Network()
    for file in data.filename.unique():
        n.add_clique(set(data.loc[data.filename==file,'author_name']))

    # remove self loops
    for edge in n.edges:
        if edge[0] == edge[1]:
            n.remove_edge(edge[0], edge[1])

    return n, node_info, edge_info


def get_bipartite_network(sqlite_db_file, time_from=None, time_to=None):
    """ Returns temporal bipartite network containing time-stamped file-author relationships for
        given time window.

        Node and edge infos set up to be expanded with future releases.

    Args:
        sqlite_db_file: path to sqlite database
        time_from: start time of time window filter, datetime object
        time_to: end time of time window filter, datetime object

    Returns:
        t: pathpy temporal network
        node_info: info on node charactaristics, e.g. membership in bipartite class
        edge_info: info on edge characteristics
    """

    con = sqlite3.connect(sqlite_db_file)
    edits = pd.read_sql("""SELECT commit_hash AS post_commit,
                                  filename
                           FROM edits""", con).drop_duplicates()

    commits = pd.read_sql("""SELECT hash, author_name, author_date AS time FROM commits""", con)

    data = pd.merge(edits, commits, how='left', left_on='post_commit', right_on='hash') \
                        .drop(['post_commit', 'hash'], axis=1)

    if time_from == None:
        time_from = datetime.datetime.strptime(min(data.time), '%Y-%m-%d %H:%M:%S')
    if time_to == None:
        time_to = datetime.datetime.strptime(max(data.time), '%Y-%m-%d %H:%M:%S')

    node_info = {}
    edge_info = {}

    node_info['class'] = {}
    t = pp.TemporalNetwork()
    for idx, row in data.iterrows():
        if (datetime.datetime.strptime(row.time, '%Y-%m-%d %H:%M:%S') >= time_from) and \
           (datetime.datetime.strptime(row.time, '%Y-%m-%d %H:%M:%S') <= time_to):
            t.add_edge(row['author_name'], row['filename'], row['time'], directed=True,
                            timestamp_format='%Y-%m-%d %H:%M:%S')
            node_info['class'][row['author_name']] = 'author'
            node_info['class'][row['filename']] = 'file'

    return t, node_info, edge_info
