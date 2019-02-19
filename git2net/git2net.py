#!/usr/bin/python3

import sqlite3
import os
import argparse
import sys

from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy

import pydriller as pydriller
from pydriller.git_repository import GitCommandError
from Levenshtein import distance as lev_dist
import datetime

import pathpy as pp
import copy
import re
import lizard


def get_block_length(lines, k):
    """
    Calculates the length (in number of lines) of a edit of added/deleted lines starting in a given
    line k.

    Parameters
    ----------
    @param lines: dictionary of added or deleted lines
    @param k: line number to check for
    """
    if k not in lines or (k > 1 and k - 1 in lines):
        edit = False
        block_size = 0
    else:
        edit = True
        block_size = 1

    while edit:
        if k + block_size in lines:
            block_size += 1
        else:
            edit = False
    return block_size


def identify_edits(deleted_lines, added_lines, use_blocks=False):
    """
    Maps line numbers between the pre- and post-commit version of a modification.
    """

    # either deleted or added lines must contain items otherwise there would not be a modification
    # to process
    if len(deleted_lines.keys()) > 0:
        max_deleted = max(deleted_lines.keys())
        min_deleted = min(deleted_lines.keys())
    else:
        max_deleted = -1
        min_deleted = np.inf

    if len(added_lines.keys()) > 0:
        max_added = max(added_lines.keys())
        min_added = min(added_lines.keys())
    else:
        max_added = -1
        min_added = np.inf

    # create mapping between pre and post edit line numbers
    pre_to_post = {}

    # create DataFrame holding information on edit
    edits = pd.DataFrame()

    # line numbers of lines before the first addition or deletion do not change
    pre = min(max(min_added, 0), max(min_deleted, 0))
    post = min(max(min_added, 0), max(min_deleted, 0))

    # counters used to match pre and post line number
    no_post_inc = 0
    both_inc = 0
    no_pre_inc = 0

    # line numbers after the last addition or deletion do not matter for edits
    while (pre <= max_deleted + 1) or (post <= max_added + 1):
        if use_blocks:
            # compute size of added and deleted edits
            # size is reported as 0 if the line is not in added or deleted lines, respectively
            length_added_block = get_block_length(added_lines, post)
            length_deleted_block = get_block_length(deleted_lines, pre)

            # replacement if both deleted and added > 0
            # if not both > 0, deletion if deleted > 0
            # if not both > 0, addition if added > 0
            if (length_deleted_block > 0) and (length_added_block > 0):
                edits = edits.append({'pre_start': int(pre),
                                      'number_of_deleted_lines': int(length_deleted_block),
                                      'post_start': int(post),
                                      'number_of_added_lines': int(length_added_block),
                                      'type': 'replacement'},
                                     ignore_index=True, sort=False)
            elif length_deleted_block > 0:
                edits = edits.append({'pre_start': int(pre),
                                      'number_of_deleted_lines': int(length_deleted_block),
                                      'post_start': int(post),
                                      'number_of_added_lines': int(length_added_block),
                                      'type': 'deletion'},
                                     ignore_index=True, sort=False)
            elif length_added_block > 0:
                edits = edits.append({'pre_start': int(pre),
                                      'number_of_deleted_lines': int(length_deleted_block),
                                      'post_start': int(post),
                                      'number_of_added_lines': int(length_added_block),
                                      'type': 'addition'},
                                     ignore_index=True, sort=False)

            # deleted edit is larger than added edit
            if length_deleted_block > length_added_block:
                no_post_inc = length_deleted_block - length_added_block
                both_inc = length_added_block
            # added edit is larger than deleted edit
            elif length_added_block > length_deleted_block:
                no_pre_inc = length_added_block - length_deleted_block
                both_inc = length_deleted_block
        else: # no blocks are considered
            pre_in_deleted = pre in deleted_lines
            post_in_added = post in added_lines
            # cf. case of blocks above
            # length of blocks is equivalent to line being in added or deleted lines
            if pre_in_deleted and post_in_added:
                edits = edits.append({'pre_start': int(pre),
                                      'number_of_deleted_lines': int(pre_in_deleted),
                                      'post_start': int(post),
                                      'number_of_added_lines': int(post_in_added),
                                      'type': 'replacement'},
                                     ignore_index=True, sort=False)
            elif pre_in_deleted and not post_in_added:
                edits = edits.append({'pre_start': int(pre),
                                      'number_of_deleted_lines': int(pre_in_deleted),
                                      'post_start': None,
                                      'number_of_added_lines': None,
                                      'type': 'deletion'},
                                     ignore_index=True, sort=False)
                no_post_inc += 1
            elif post_in_added and not pre_in_deleted:
                edits = edits.append({'pre_start': None,
                                      'number_of_deleted_lines': None,
                                      'post_start': int(post),
                                      'number_of_added_lines': int(post_in_added),
                                      'type': 'addition'},
                                     ignore_index=True, sort=False)
                no_pre_inc += 1

        # increment pre and post counter
        if both_inc > 0:
            both_inc -= 1
            pre_to_post[pre] = post
            pre += 1
            post += 1
        elif no_post_inc > 0:
            no_post_inc -= 1
            pre_to_post[pre] = False
            pre += 1
        elif no_pre_inc > 0:
            no_pre_inc -= 1
            post += 1
        else:
            pre_to_post[pre] = post
            pre += 1
            post += 1

    return pre_to_post, edits


def text_entropy(text):
    return entropy([text.count(chr(i)) for i in range(256)], base=2)


def get_commit_dag(repo_string):
    git_repo = pydriller.GitRepository(repo_string)
    commits = [x.hash[0:7] for x in git_repo.get_list_commits()]
    dag = pp.DAG()
    for node in commits:
        for parent in git_repo.get_commit(node).parents:
            dag.add_edge(parent[0:7], node)
    return dag


def parse_blame_C(blame_C):
    pattern = re.compile("(^$|^-?C{0,3}[0-9]*$)")
    if not pattern.match(blame_C):
        raise Exception("Invalid 'blame_C' supplied.")
    if len(blame_C) == 0:
        return []
    else:
        if blame_C[0] == '-':
            blame_C = blame_C[1:]
        cs = len(blame_C) - len(blame_C.lstrip('C'))
        num = blame_C.lstrip('C')
        return ['-C' for i in range(cs - 1)] + ['-C{}'.format(num)]


def parse_porcelain_blame(blame):
    l = {'original_commit_hash': [],
        'original_line_no': [],
        'original_file_path': [],
        'line_content': [],
        'line_number': []}
    start_of_line_info = True
    prefix = '\t'
    line_number = 1
    for idx, line in enumerate(blame.split('\n')):
        if line.startswith(prefix):
            l['original_file_path'].append(filename)
            l['line_content'].append(line[len(prefix):])
            l['line_number'].append(line_number)
            line_number += 1
            start_of_line_info = True
        else:
            entries = line.split(' ')
            if start_of_line_info:
                l['original_commit_hash'].append(entries[0])
                l['original_line_no'].append(entries[1])
                start_of_line_info = False
            elif entries[0] == 'filename':
                filename = entries[1]
    return pd.DataFrame(l)

def get_edit_details(edit, commit, deleted_lines, added_lines, blame_info_parent, blame_info_commit,
                     use_blocks=False):
    # Different actions for different types of edits.
    e = {}
    if edit.type == 'replacement':
        # For replacements, both the content of the deleted and added block are required in
        # order to compute text entropy, as well as Levenshtein edit distance between them.
        deleted_block = []
        for i in range(int(edit.pre_start), int(edit.pre_start + edit.number_of_deleted_lines)):
            deleted_block.append(deleted_lines[i])

        added_block = []
        for i in range(int(edit.post_start), int(edit.post_start + edit.number_of_added_lines)):
            added_block.append(added_lines[i])

        # For the analysis, lines are concatenated with whitespaces.
        deleted_block = ' '.join(deleted_block)
        added_block = ' '.join(added_block)

        # Given this, all metadata can be written.
        # Data on the content and location of deleted line in the parent commit.
        e['pre_starting_line_no'] = int(edit.pre_start)
        e['pre_len_in_lines'] = int(edit.number_of_deleted_lines)
        e['pre_len_in_chars'] = len(deleted_block)
        e['pre_entropy'] = text_entropy(deleted_block)

        # Data on the content and location of added line in the current commit.
        e['post_starting_line_no'] = int(edit.post_start)
        e['post_len_in_lines'] = int(edit.number_of_added_lines)
        e['post_len_in_chars'] = len(added_block)
        e['post_entropy'] = text_entropy(added_block)

        # Levenshtein edit distance between deleted and added block.
        e['levenshtein_dist'] = lev_dist(deleted_block, added_block)

        # Data on origin of deleted line. Every deleted line must have an origin
        if use_blocks:
            e['original_commit_deletion'] = 'not available with use_blocks'
            e['original_line_no_deletion'] = 'not available with use_blocks'
            e['original_file_path_deletion'] = 'not available with use_blocks'
        else:
            assert blame_info_parent is not None
            e['original_commit_deletion'] = blame_info_parent.at[int(edit.pre_start) - 1,
                                                                    'original_commit_hash']
            e['original_line_no_deletion'] = blame_info_parent.at[int(edit.pre_start) - 1,
                                                                    'original_line_no']
            e['original_file_path_deletion'] = blame_info_parent.at[int(edit.pre_start) - 1,
                                                                    'original_file_path']

        # Data on the origin of added line. Can be either original or copied form other file.
        if use_blocks:
            e['original_commit_addition'] = 'not available with use_blocks'
            e['original_line_no_addition'] = 'not available with use_blocks'
            e['original_file_path_addition'] = 'not available with use_blocks'
        elif blame_info_commit.at[int(edit.post_start) - 1,
                                    'original_commit_hash'] == commit.hash:
            # The line is original, there exists no original commit, line number or file path.
            e['original_commit_addition'] = None
            e['original_line_no_addition'] = None
            e['original_file_path_addition'] = None
        else:
            # The line was copied from somewhere.
            assert blame_info_commit is not None
            e['original_commit_addition'] = blame_info_commit.at[int(edit.post_start) - 1,
                                                                    'original_commit_hash']
            e['original_line_no_addition'] = blame_info_commit.at[int(edit.post_start) - 1,
                                                                    'original_line_no']
            e['original_file_path_addition'] = blame_info_commit.at[int(edit.post_start) - 1,
                                                                    'original_file_path']

    elif edit.type == 'deletion':
        # For deletions, only the content of the deleted block is required.
        deleted_block = []
        for i in range(int(edit.pre_start), int(edit.pre_start + edit.number_of_deleted_lines)):
            deleted_block.append(deleted_lines[i])

        deleted_block = ' '.join(deleted_block)

        # Given this, all metadata can be written.
        # Data on the deleted line in the parent commit.
        e['pre_starting_line_no'] = int(edit.pre_start)
        e['pre_len_in_lines'] = int(edit.number_of_deleted_lines)
        e['pre_len_in_chars'] = len(deleted_block)
        e['pre_entropy'] = text_entropy(deleted_block)

        # For deletions, there is no added line.
        e['post_starting_line_no'] = None
        e['post_len_in_lines'] = None
        e['post_len_in_chars'] = None
        e['post_entropy'] = None
        e['original_commit_addition'] = None
        e['original_line_no_addition'] = None
        e['original_file_path_addition'] = None

        # Levenshtein edit distance is set to 'None'. Theoretically 1 keystroke required.
        e['levenshtein_dist'] = None

        # Data on origin of deleted line. Every deleted line must have an origin
        if use_blocks:
            e['original_commit_deletion'] = 'not available with use_blocks'
            e['original_line_no_deletion'] = 'not available with use_blocks'
            e['original_file_path_deletion'] = 'not available with use_blocks'
        else:
            assert blame_info_parent is not None
            e['original_commit_deletion'] = blame_info_parent.at[int(edit.pre_start) - 1,
                                                                    'original_commit_hash']
            e['original_line_no_deletion'] = blame_info_parent.at[int(edit.pre_start) - 1,
                                                                    'original_line_no']
            e['original_file_path_deletion'] = blame_info_parent.at[int(edit.pre_start) - 1,
                                                                    'original_file_path']

    elif edit.type == 'addition':
        # For additions, only the content of the added block is required.
        added_block = []
        for i in range(int(edit.post_start), int(edit.post_start + edit.number_of_added_lines)):
            added_block.append(added_lines[i])

        added_block = ' '.join(added_block)

        # Given this, all metadata can be written.
        # For additions, there is no deleted line.
        e['pre_starting_line_no'] = None
        e['pre_len_in_lines'] = None
        e['pre_len_in_chars'] = None
        e['pre_entropy'] = None
        e['original_commit_deletion'] = None
        e['original_line_no_deletion'] = None
        e['original_file_path_deletion'] = None

        # Data on the added line.
        e['post_starting_line_no'] = int(edit.post_start)
        e['post_len_in_lines'] = int(edit.number_of_added_lines)
        e['post_len_in_chars'] = len(added_block)
        e['post_entropy'] = text_entropy(added_block)

        # Levenshtein edit distance is length of added block as nothing existed before.
        e['levenshtein_dist'] = len(added_block)

        # If the lines were newly added to this file, they might still come from another file.
        if use_blocks:
            e['original_commit_addition'] = 'not available with use_blocks'
            e['original_line_no_addition'] = 'not available with use_blocks'
            e['original_file_path_addition'] = 'not available with use_blocks'
        elif blame_info_commit.at[int(edit.post_start) - 1,
                                    'original_commit_hash'] == commit.hash:
            # The line is original, there exists no original commit, line number or file path.
            e['original_commit_addition'] = None
            e['original_line_no_addition'] = None
            e['original_file_path_addition'] = None
        else:
            # The line was copied from somewhere.
            assert blame_info_commit is not None
            e['original_commit_addition'] = blame_info_commit.at[int(edit.post_start) - 1,
                                                                    'original_commit_hash']
            e['original_line_no_addition'] = blame_info_commit.at[int(edit.post_start) - 1,
                                                                    'original_line_no']
            e['original_file_path_addition'] = blame_info_commit.at[int(edit.post_start) - 1,
                                                                    'original_file_path']

    elif edit.type == 'file_renaming':
        # For file renaming only old and new path are required which were already set before.
        e['pre_starting_line_no'] = None
        e['pre_len_in_lines'] = None
        e['pre_len_in_chars'] = None
        e['pre_entropy'] = None
        e['post_starting_line_no'] = None
        e['post_len_in_lines'] = None
        e['post_len_in_chars'] = None
        e['post_entropy'] = None

        if use_blocks:
            e['original_commit_deletion'] = 'not available with use_blocks'
            e['original_line_no_deletion'] = 'not available with use_blocks'
            e['original_file_path_deletion'] = 'not available with use_blocks'
            e['original_commit_addition'] = 'not available with use_blocks'
            e['original_line_no_addition'] = 'not available with use_blocks'
            e['original_file_path_addition'] = 'not available with use_blocks'
        else:
            e['original_commit_deletion'] = None
            e['original_line_no_deletion'] = None
            e['original_file_path_deletion'] = None
            e['original_commit_addition'] = None
            e['original_line_no_addition'] = None
            e['original_file_path_addition'] = None

        # Levenshtein edit distance set to 0 to distinguish from deletion
        e['levenshtein_dist'] = 0

    else:
        print(edit.type)
        raise Exception("Unexpected error in 'get_edit_details'.")

    return e

def extract_edits(git_repo, commit, modification, use_blocks=False, blame_C='-C'):
    """Returns dataframe with metadata on edits made in a given modification.

    Args:
        git_repo: pydriller GitRepository object
        commit: pydriller Commit object
        modification: pydriller Modification object
        use_blocks: bool, determins if analysis is performed on block or line basis
        blame_C: git blame '-C' option. By default, '-C' is used.

    Returns:
        edits_info: pandas DataFrame object containing metadata on all edits in given modification
    """

    # Parse diff of given modification to extract added and deleted lines
    parsed_lines = git_repo.parse_diff(modification.diff)

    deleted_lines = { x[0]:x[1] for x in parsed_lines['deleted'] }
    added_lines = { x[0]:x[1] for x in parsed_lines['added'] }

    # If there was a modification but no lines were added or removed, the file was renamed.
    if (len(deleted_lines) == 0) and (len(added_lines) == 0):
        edits = pd.DataFrame({'pre_start': None,
                              'number_of_deleted_lines': None,
                              'post_start': None,
                              'number_of_added_lines': None,
                              'type': 'file_renaming'}, index=[0])
    else: # If there were lines added or deleted, the specific edits are identified.
        _, edits = identify_edits(deleted_lines, added_lines, use_blocks=use_blocks)

    # In order to trace the origins of lines e execute git blame is executed. For lines that were
    # deleted with the current commit, the blame needs to be executed on the parent commit. As
    # merges are treated separately, commits should only have one parent. For added lines, git blame
    # is executed on the current commit.
    blame_info_parent = None
    blame_info_commit = None
    if len(deleted_lines) > 0:
        assert len(commit.parents) == 1
        blame_parent = git_repo.git.blame(commit.parents[0],
                                          parse_blame_C(blame_C) +
                                          ['-w', '--show-number', '--porcelain'],
                                          modification.old_path)
        blame_info_parent = parse_porcelain_blame(blame_parent)

    if len(added_lines) > 0:
        blame_commit = git_repo.git.blame(commit.hash,
                                          parse_blame_C(blame_C) +
                                          ['-w', '--show-number', '--porcelain'],
                                          modification.new_path)
        blame_info_commit = parse_porcelain_blame(blame_commit)

    # Next, metadata on all identified edits is extracted and added to a pandas DataFrame.
    edits_info = pd.DataFrame()
    for _, edit in tqdm(edits.iterrows(), leave=False, desc='edits', total=len(edits)):
        e = {}
        # Extract general information.
        e['filename'] = modification.filename
        e['new_path'] = modification.new_path
        e['old_path'] = modification.old_path
        e['commit_hash'] = commit.hash
        e['total_added_lines'] = modification.added
        e['total_removed_lines'] = modification.removed
        e['cyclomatic_complexity_of_file'] = modification.complexity
        e['lines_of_code_in_file'] = modification.nloc
        e['modification_type'] = modification.change_type.name
        e['edit_type'] = edit.type

        e.update(get_edit_details(edit, commit, deleted_lines, added_lines, blame_info_parent,
                                  blame_info_commit))

        edits_info = edits_info.append(e, ignore_index=True, sort=False)

    return edits_info

def extract_edits_merge(git_repo, commit, modification_info, use_blocks=False, blame_C='-C'):
    assert commit.merge
    # With merges, the following cases can occur:
    #   1. Changes of one or more parents are accepted.
    #   2. Changes made prior to the merge are replaced with new edits.
    # To obtain the state of the file before merging, get blame is executed on all parent commits.
    parent_blames = []
    for parent in commit.parents:
        blame = git_repo.git.blame(parent,
                                   parse_blame_C(blame_C) +
                                   ['-w', '--show-number', '--porcelain'],
                                   modification_info['old_path'])
        parent_blame = parse_porcelain_blame(blame).rename(
                    columns={'line_content': 'pre_line_content', 'line_number': 'pre_line_number'})
        parent_blame.loc[:, 'pre_commit'] = parent
        parent_blames.append(parent_blame)

    # Define columns that are considered when identifying duplicates.
    comp_cols = ['original_commit_hash', 'original_line_no', 'original_file_path']

    # Differences between parents are obtained by concatenating and fully dropping all duplicates.
    parent_differences = pd.concat(parent_blames).drop_duplicates(subset=comp_cols, keep=False)
    parents_equality = pd.concat(parent_blames).loc[pd.concat(parent_blames).duplicated(
                                                                            subset=comp_cols), :]
    parents_equality.drop(['pre_commit'], axis=1, inplace=True)

    # Then, the current state of the file is obtained by executing git blame on the current commit.
    blame = git_repo.git.blame(commit.hash,
                               parse_blame_C(blame_C) +
                               ['-w', '--show-number', '--porcelain'],
                               modification_info['new_path'])
    current_blame = parse_porcelain_blame(blame).rename(
            columns={'line_content': 'post_line_content', 'line_number': 'post_line_number'})
    # Lines that are in both parents but not in current state were actively deleted in the merge.
    deleted = parents_equality.merge(current_blame, on=comp_cols, how='left', indicator=True)
    deleted = deleted.loc[deleted._merge != 'both', :].drop(['_merge'], axis=1)
    deleted.drop(['post_line_content', 'post_line_number'], axis=1, inplace=True)

    # All lines for which git blame shows they originate in the current commit were actively added.
    added = current_blame.loc[current_blame.original_commit_hash == commit.hash, :]

    # Now, only the lines that in the current commit that were not added or in both parents remain.
    remaining = pd.concat([current_blame, parents_equality, added], sort=False).drop_duplicates(
                                                                    subset=comp_cols, keep=False)
    remaining.drop(['pre_line_content', 'pre_line_number'], axis=1, inplace=True)

    # These lines have to come from the difference between the parents. All lines that are not in
    # this difference were removed by not accepting changes in the merge.
    deleted_merge = parent_differences.merge(remaining, on=comp_cols, how='left', indicator=True)
    deleted_merge = deleted_merge.loc[deleted_merge._merge != 'both', :].drop(['_merge'], axis=1)
    deleted_merge.drop(['post_line_content', 'post_line_number'], axis=1, inplace=True)

    # When lines are added, they can replace lines that were deleted with the same merge. These are
    # identified next.
    edits = pd.DataFrame()
    edits_info = pd.DataFrame()
    for idx, parent in enumerate(commit.parents):
        deleted_p = pd.concat([deleted, deleted_merge.loc[deleted_merge.pre_commit == parent,:]
                               .drop(['pre_commit'], axis=1)], sort=False)

        added_lines = {x.post_line_number: x.post_line_content for _, x in added.iterrows()}
        deleted_lines = {x.pre_line_number: x.pre_line_content for _, x in deleted_p.iterrows()}

        _, edits = identify_edits(deleted_lines, added_lines, use_blocks=use_blocks)

        for _, edit in tqdm(edits.iterrows(), leave=False, desc='edits', total=len(edits)):
            e = {}
            # Extract general information.
            e['commit_hash'] = commit.hash
            e['edit_type'] = edit.type
            e.update(modification_info)
            e.update(get_edit_details(edit, commit, deleted_lines, added_lines, parent_blames[idx],
                                    current_blame, use_blocks=use_blocks))

            edits_info = edits_info.append(e, ignore_index=True, sort=False)

    return edits_info


def get_edited_file_paths_since_split(git_repo, commit):
    def expand_dag(dag, leafs):
        for node in leafs:
            parents = git_repo.get_commit(node).parents
            for parent in parents:
                dag.add_edge(node, parent)
        return dag
    def common_node_on_paths(paths):
        # Drop first and last element of the path.
        common_nodes = set(paths[0][1:-1])
        for path in paths[1:]:
            common_nodes.intersection_update(path[1:-1])
        return list(common_nodes)

    def remove_successors(dag, node):
        rm = [n for nl in [x[1:] for x in dag.routes_from_node(node)] for n in nl]
        for node in rm:
            dag.remove_node(node)
        return dag

    dag = pp.DAG()
    dag.add_node(commit.hash)

    leafs = list(dag.nodes)

    cont = True
    while cont:
        dag = expand_dag(dag, leafs)
        leafs = [node for node in dag.nodes if len(dag.successors[node]) == 0]

        paths = [p for pl in [dag.routes_to_node(node) for node in leafs] for p in pl]
        common_nodes = common_node_on_paths(paths)

        if (len(leafs) == 1) or (len(common_nodes) > 0):
            cont = False

    if len(common_nodes) > 0:
        dag = remove_successors(dag, common_nodes[0])

    edited_file_paths = []
    for node in dag.nodes:
        edited_file_paths += [modification.new_path for modification
                              in git_repo.get_commit(node).modifications]
        edited_file_paths += [modification.old_path for modification
                              in git_repo.get_commit(node).modifications]

    edited_file_paths = set(edited_file_paths)
    if None in edited_file_paths:
        edited_file_paths.remove(None)

    return edited_file_paths


def process_commit(args):
    git_repo = pydriller.GitRepository(args['repo_string'])
    commit = git_repo.get_commit(args['commit_hash'])

    # parse commit
    c = {}
    c['hash'] = commit.hash
    c['author_email'] = commit.author.email
    c['author_name'] = commit.author.name
    c['committer_email'] = commit.committer.email
    c['committer_name'] = commit.committer.name
    c['author_date'] = commit.author_date.strftime('%Y-%m-%d %H:%M:%S')
    c['committer_date'] = commit.committer_date.strftime('%Y-%m-%d %H:%M:%S')
    c['committer_timezone'] = commit.committer_timezone
    c['no_of_modifications'] = len(commit.modifications)
    c['msg_len'] = len(commit.msg)
    c['project_name'] = commit.project_name
    c['parents'] = ','.join(commit.parents)
    c['merge'] = commit.merge
    c['in_main_branch'] = commit.in_main_branch
    c['branches'] = ','.join(commit.branches)

    # parse modification
    df_edits = pd.DataFrame()
    if commit.merge:
        # Git does not create a modification if own changes are accpeted during a merge. Therefore,
        # the edited files are extracted manually.
        edited_file_paths = get_edited_file_paths_since_split(git_repo, commit)
        for edited_file_path in edited_file_paths:
            exclude_file = False
            for x in args['exclude_paths']:
                if edited_file_path.startswith(x + os.sep) or (edited_file_path == x):
                    exclude_file = True
            if not exclude_file:
                modification_info = {}
                try:
                    file_contents = git_repo.git.show('{}:{}'.format(commit.hash, edited_file_path))
                except GitCommandError:
                    # A GitCommandError occurs if the file was deleted. In this case it currently
                    # has no content.
                    file_contents = ''
                l = lizard.analyze_file.analyze_source_code(edited_file_path, file_contents)

                modification_info['filename'] = edited_file_path.split(os.sep)[-1]
                modification_info['new_path'] = edited_file_path
                modification_info['old_path'] = edited_file_path
                modification_info['cyclomatic_complexity_of_file'] = l.CCN
                modification_info['lines_of_code_in_file'] = l.nloc
                modification_info['modification_type'] = 'merge_self_accept'

                df_edits = df_edits.append(extract_edits_merge(git_repo, commit,
                                                                modification_info,
                                                                use_blocks=args['use_blocks'],
                                                                blame_C=args['blame_C']),
                                            ignore_index=True, sort=True)
    else:
        for modification in tqdm(commit.modifications, leave=False, desc='modifications'):
            exclude_file = False
            for x in args['exclude_paths']:
                if modification.new_path:
                    if modification.new_path.startswith(x + os.sep) or (modification.new_path == x):
                        exclude_file = True
                if not exclude_file and modification.old_path:
                    if modification.old_path.startswith(x + os.sep):
                        exclude_file = True
            if not exclude_file:
                df_edits = df_edits.append(extract_edits(git_repo, commit, modification,
                                                         use_blocks=args['use_blocks'],
                                                         blame_C=args['blame_C']),
                                           ignore_index=True, sort=True)


    df_commit = pd.DataFrame(c, index=[0])

    return {'commit': df_commit, 'edits': df_edits}


def process_repo_serial(repo_string, sqlite_db_file, use_blocks=False, exclude=None, blame_C='-C',
                        _p_commits=[]):
    git_repo = pydriller.GitRepository(repo_string)
    exclude_paths = []
    if exclude:
        with open(exclude) as f:
            exclude_paths = [x.strip() for x in f.readlines()]

    df_commits = pd.DataFrame()
    df_edits = pd.DataFrame()

    commits = [commit for commit in git_repo.get_list_commits() if commit.hash not in _p_commits]

    con = sqlite3.connect(sqlite_db_file)

    for commit in tqdm(commits, desc='Serial'):
        args = {'repo_string': repo_string, 'commit_hash': commit.hash, 'use_blocks': use_blocks,
                'exclude_paths': exclude_paths, 'blame_C': blame_C}
        result = process_commit(args)

        if not result['commit'].empty:
            result['commit'].to_sql('commits', con, if_exists='append')
        if not result['edits'].empty:
            result['edits'].to_sql('edits', con, if_exists='append')


def process_repo_parallel(repo_string, sqlite_db_file, use_blocks=False,
                          no_of_processes=os.cpu_count(), chunksize=1, exclude=None, blame_C='-C',
                          _p_commits=[]):
    git_repo = pydriller.GitRepository(repo_string)
    exclude_paths = []
    if exclude:
        with open(exclude) as f:
            exclude_paths = [x.strip() for x in f.readlines()]

    args = [{'repo_string': repo_string, 'commit_hash': commit.hash, 'use_blocks': use_blocks,
             'exclude_paths': exclude_paths, 'blame_C': blame_C}
            for commit in git_repo.get_list_commits() if commit.hash not in _p_commits]

    con = sqlite3.connect(sqlite_db_file)
    p = Pool(no_of_processes)

    with tqdm(total=len(args), desc='Parallel ({0} processes)'.format(no_of_processes)) as pbar:
        for result in p.imap_unordered(process_commit, args, chunksize=chunksize):
            if not result['commit'].empty:
                result['commit'].to_sql('commits', con, if_exists='append')
            if not result['edits'].empty:
                result['edits'].to_sql('edits', con, if_exists='append')
            pbar.update(1)


def extract_editing_paths(sqlite_db_file, commit_hashes=False, file_paths=False, with_start=False,
                          merge_renaming=True):
    """ Returns line editing DAG as well as line editing paths.

    Args:
        sqlite_db_file: path to sqlite database mined with git2net line method
        commit_hashes: list of commits to consider, by default all commits are considered
        file_paths: list of files to consider, by defailt all files are considered
        with_start: bool, determines if node for filename is included as start for all editing pahts
        merge_renaming: bool, determines if file renaming is considered

    Returns:
        dag: line editing directed acyclic graph, pathpy DAG object
        paths: line editing pahts, pathpy Path object
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

    # commits.loc[:, 'hash'] = commits.hash.apply(lambda x: x[0:7] if x is not None else x)

    # edits.loc[:, 'pre_commit'] = edits.pre_commit.apply(lambda x: x[0:7] if x is not None else x)
    # edits.loc[:, 'post_commit'] = edits.post_commit.apply(
    #                                 lambda x: x[0:7] if x is not None else x)
    # edits.loc[:, 'original_line_no'] = edits.original_line_no.apply(
    #                                             lambda x: float(x) if x is not None else x)

    # Filter edits table if only edits from specific commits are considered.
    if not commit_hashes == False:
        edits = edits.loc[[x in commit_hashes for x in edits.commit_hash], :]

    # Rename file paths to latest name if option is selected.
    if merge_renaming:
        # Identify files that have been renamed.
        _, aliases = identify_file_renaming(path)
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
            file_paths.add(edit.new_path)
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
        elif '#FBB13C' in [edge_info['colors'][n] for n in [(x, node) for x in dag.predecessors[node]]]:
            node_info['colors'][node] = '#FBB13C' # yellow
        elif node.startswith('deleted'):
            node_info['colors'][node] = '#A8322D' # red
        elif 'white' not in [edge_info['colors'][n] for n in [(node, x) for x in dag.successors[node]]]:
            node_info['colors'][node] = '#2E5EAA' # blue
        elif not dag.predecessors[node].isdisjoint(file_paths):
            node_info['colors'][node] = '#218380' # green
        else:
            node_info['colors'][node] = '#73D2DE' # light blue
    #     elif node.startswith('deleted line'):
    #         node_info['colors'][node] = '#A8322D' # red
    #     elif (file_path in dag.predecessors[node]) and \
    #          (len(dag.successors[node]) == 0):
    #         node_info['colors'][node] = '#5B4E77' # purple
    #     elif file_path in dag.predecessors[node]:
    #         node_info['colors'][node] = '#218380' # green
    #     elif len(dag.successors[node]) == 0:
    #         node_info['colors'][node] = '#2E5EAA' # blue
    #     else:
    #         node_info['colors'][node] = '#73D2DE' # light blue


    if not with_start:
        for file_path in file_paths:
            dag.remove_node(file_path)

    #assert dag.is_acyclic is True

    paths = pp.path_extraction.paths_from_dag(dag)

    return dag, paths, node_info, edge_info


def identify_file_renaming(repo_string):
    git_repo = pydriller.GitRepository(repo_string)

    dag = pp.DAG()
    for commit in git_repo.get_list_commits():
        for modification in commit.modifications:
            if (modification.new_path not in dag.nodes) and \
               (modification.old_path == modification.new_path) and \
               (modification.change_type == pydriller.domain.commit.ModificationType.ADD):
                dag.add_edge('added file', modification.new_path)
            elif modification.old_path != modification.new_path:
                if pd.isnull(modification.old_path):
                    dag.add_edge('added file', modification.new_path)
                elif pd.isnull(modification.new_path):
                    dag.add_edge(modification.old_path, 'deleted file')
                else:
                    dag.add_edge(modification.old_path, modification.new_path)

    def get_path_to_leaf_node(dag, node, _path=[]):
        """
        returns path to leaf node with leaf node at position '0' in list
        """
        if len(dag.successors[node]) > 0:
            return get_path_to_leaf_node(dag, list(dag.successors[node])[0], _path=[node] + _path)
        else:
            return [node] + _path

    renamings = []
    for node in dag.nodes:
        if 'added file' in dag.predecessors[node]:
            renamings.append(get_path_to_leaf_node(dag, node))

    aliases = {}
    for renaming in renamings:
        if 'deleted file' in renaming:
            renaming.remove('deleted file')
        for alias in renaming:
            aliases[alias] = renaming

    return dag, aliases


def get_unified_changes(repo_string, commit_hash, file_path):
    """
    Returns dataframe with github-like unified diff representation of the content of a file before
    and after a commit for a given git repository, commit hash and file path.
    """
    git_repo = pydriller.GitRepository(repo_string)
    commit = git_repo.get_commit(commit_hash)
    for modification in commit.modifications:
        if modification.file_path == file_path:
            parsed_lines = git_repo.parse_diff(modification.diff)

            deleted_lines = { x[0]:x[1] for x in parsed_lines['deleted'] }
            added_lines = { x[0]:x[1] for x in parsed_lines['added'] }

            pre_to_post, edits = identify_edits(deleted_lines, added_lines)

            post_source_code = modification.source_code.split('\n')

            max_line_no = max(max(deleted_lines.keys()),
                              max(added_lines.keys()),
                              len(post_source_code))

            pre_line_no = []
            post_line_no = []
            action = []
            code = []

            pre_counter = 1
            post_counter = 1
            while max(pre_counter, post_counter) < max_line_no:
                if pre_counter in edits.keys():
                    cur = pre_counter
                    for i in range(edits[cur][0]):
                        pre_line_no.append(pre_counter)
                        post_line_no.append(None)
                        action.append('-')
                        code.append(deleted_lines[pre_counter])
                        pre_counter += 1
                    for i in range(edits[cur][2]):
                        pre_line_no.append(None)
                        post_line_no.append(post_counter)
                        action.append('+')
                        code.append(added_lines[post_counter])
                        post_counter += 1
                else:
                    if pre_counter in pre_to_post.keys():
                        # if pre is not in the dictionary nothing has changed
                        if post_counter < pre_to_post[pre_counter]:
                            # edit has been added
                            for i in range(pre_to_post[pre_counter] - post_counter):
                                pre_line_no.append(None)
                                post_line_no.append(post_counter)
                                action.append('+')
                                code.append(added_lines[post_counter])
                                post_counter += 1

                    pre_line_no.append(pre_counter)
                    post_line_no.append(post_counter)
                    action.append(None)
                    code.append(post_source_code[post_counter - 1]) # -1 as list starts from 0
                    pre_counter += 1
                    post_counter += 1

    return pd.DataFrame({'pre': pre_line_no, 'post': post_line_no, 'action': action, 'code': code})


# THIS FUNCTIONS NEEDS TO BE REWRITTEN BASED ON THE NEW RESULTS

# def _get_tedges(db_location):
#     con = sqlite3.connect(db_location)

#     tedges = pd.read_sql("""SELECT x.author_pre as source,
#                                    substr(x.pre_commit, 1, 8) as pre_commit,
#                                    c_post.author_email AS target,
#                                    substr(x.post_commit, 1, 8) AS post_commit,
#                                    c_post.committer_date as time,
#                                    x.levenshtein_dist as levenshtein_dist
#                             FROM (
#                                    SELECT c_pre.author_email AS author_pre,
#                                           edits.pre_commit,
#                                           edits.post_commit,
#                                           edits.levenshtein_dist
#                                    FROM edits
#                                    JOIN commits AS c_pre
#                                    ON substr(c_pre.hash, 1, 8) == edits.pre_commit) AS x
#                                    JOIN commits AS c_post
#                                    ON substr(c_post.hash, 1, 8) == substr(x.post_commit, 1, 8
#                                  )
#                             WHERE source != target""", con)

#     tedges.loc[:,'time'] = pd.to_datetime(tedges.time)

#     return tedges


# def get_coediting_network(db_location, time_from=None, time_to=None):
#     tedges = _get_tedges(db_location)

#     if time_from == None:
#         time_from = min(tedges.time)
#     if time_to == None:
#         time_to = max(tedges.time)

#     t = pp.TemporalNetwork()
#     for _, edge in tedges.iterrows():
#         if (edge.time >= time_from) and (edge.time <= time_to):
#             t.add_edge(edge.source,
#                        edge.target,
#                        edge.time.strftime('%Y-%m-%d %H:%M:%S'),
#                        directed=True,
#                        timestamp_format='%Y-%m-%d %H:%M:%S')
#     return t


# def _get_bipartite_edges(db_location):
#     con = sqlite3.connect(db_location)

#     bipartite_edges = pd.read_sql("""SELECT DISTINCT mod_filename AS target,
#                                             commits.author_name AS source,
#                                             commits.committer_date AS time
#                                      FROM edits
#                                      JOIN commits ON edits.post_commit == commits.hash""", con)

#     bipartite_edges.loc[:,'time'] = pd.to_datetime(bipartite_edges.time)

#     return bipartite_edges


# def get_bipartite_network(db_location, time_from=None, time_to=None):
#     bipartite_edges = _get_bipartite_edges(db_location)

#     if time_from == None:
#         time_from = min(bipartite_edges.time)
#     if time_to == None:
#         time_to = max(bipartite_edges.time)

#     dag = pp.Network()
#     for idx, edge in bipartite_edges.iterrows():
#         if (edge.time >= time_from) and (edge.time <= time_to):
#             dag.add_edge(edge.source, edge.target)
#     return dag


# def _get_dag_edges(db_location):
#     con = sqlite3.connect(db_location)

#     dag_edges = pd.read_sql("""SELECT DISTINCT x.author_pre||","||substr(x.pre_commit, 1, 8)
#                                         AS source,
#                                       c_post.author_email||","|| substr(x.post_commit, 1, 8)
#                                         AS target,
#                                       c_post.committer_date AS time
#                                FROM (
#                                       SELECT c_pre.author_email AS author_pre,
#                                              edits.pre_commit,
#                                              edits.post_commit,
#                                              edits.levenshtein_dist
#                                       FROM edits
#                                       JOIN commits AS c_pre
#                                       ON substr(c_pre.hash, 1, 8) == edits.pre_commit
#                                     ) AS x
#                                 JOIN (
#                                        SELECT *
#                                        FROM commits
#                                      ) AS c_post
#                                 ON substr(c_post.hash, 1, 8) == substr(x.post_commit, 1, 8)
#                                 WHERE x.author_pre != c_post.author_email""", con)

#     dag_edges.loc[:,'time'] = pd.to_datetime(dag_edges.time)

#     return dag_edges


# def get_dag(db_location, time_from=None, time_to=None):
#     dag_edges = _get_dag_edges(db_location)

#     if time_from == None:
#         time_from = min(dag_edges.time)
#     if time_to == None:
#         time_to = max(dag_edges.time)

#     dag = pp.DAG()
#     for _, edge in dag_edges.iterrows():
#         if (edge.time >= time_from) and (edge.time <= time_to):
#             dag.add_edge(edge.source, edge.target)

#     dag.topsort()

#     return dag


def mine_git_repo(repo_string, sqlite_db_file, use_blocks=False,
                  no_of_processes=os.cpu_count(), chunksize=1, exclude=[], blame_C='-C'):

    if os.path.exists(sqlite_db_file):
        try:
            with sqlite3.connect(sqlite_db_file) as con:
                prev_method, prev_repository = con.execute("""SELECT method, repository
                                                              FROM _metadata""").fetchall()[0]

                if (prev_method == 'blocks' if use_blocks else 'lines') and \
                   (prev_repository == repo_string):
                    try:
                        p_commits = set(x[0]
                            for x in con.execute("SELECT hash FROM commits").fetchall())
                    except sqlite3.OperationalError:
                        p_commits = set()
                    c_commits = set(c.hash
                        for c in pydriller.GitRepository(repo_string).get_list_commits())
                    if not p_commits.issubset(c_commits):
                        raise Exception("Found a database that was created with identical " +
                                        "settings. However, some commits in the database are not " +
                                        "in the provided git repository. Please provide a clean " +
                                        "database.")
                    else:
                        if p_commits == c_commits:
                            print("The provided database is already complete!")
                            return
                        else:
                            print("Found a matching database on provided path. " +
                                    "Skipping {} ({:.2f}%) of {} commits. {} commits remaining."
                                    .format(len(p_commits), len(p_commits) / len(c_commits) * 100,
                                            len(c_commits), len(c_commits) - len(p_commits)))
                else:
                    raise Exception("Found a database on provided path that was created with " +
                                    "settings not matching the ones selected for the current " +
                                    "run. A path to either no database or a database from a  " +
                                    "previously paused run with identical settings is required.")
        except sqlite3.OperationalError:
            raise Exception("Found a database on provided path that was likely not created with " +
                            "git2net. A path to either no database or a database from a " +
                            "previously paused run with identical settings is required.")
    else:
        print("Found no database on provided path. Starting from scratch.")
        with sqlite3.connect(sqlite_db_file) as con:
            con.execute("""CREATE TABLE _metadata ('created with',
                                                   'repository',
                                                   'date',
                                                   'method')""")
            con.execute("""INSERT INTO _metadata ('created with',
                                                  'repository',
                                                  'date',
                                                  'method')
                        VALUES (:version,
                                :repository,
                                :date,
                                :method)""",
                        {'version': 'git2net alpha',
                         'repository': repo_string,
                         'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                         'method': 'blocks' if use_blocks else 'lines'})
            con.commit()
            p_commits = []

    if no_of_processes > 1:
        process_repo_parallel(repo_string=repo_string, sqlite_db_file=sqlite_db_file,
                              use_blocks=use_blocks, no_of_processes=no_of_processes,
                              chunksize=chunksize, exclude=exclude, blame_C=blame_C,
                              _p_commits=p_commits)
    else:
        process_repo_serial(repo_string=repo_string, sqlite_db_file=sqlite_db_file,
                            use_blocks=use_blocks, exclude=exclude, blame_C=blame_C,
                            _p_commits=p_commits)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extracts commit and co-editing data from git repositories.')
    parser.add_argument('repo', help='Path to repository to be parsed.', type=str)
    parser.add_argument('outfile', help='Path to SQLite DB file storing results.', type=str)
    parser.add_argument('--use-blocks',
        help='Compare added and deleted blocks of code rather than lines.', dest='use_blocks',
        action='store_true', default=False)
    parser.add_argument('--numprocesses',
        help='Number of CPU cores used for multi-core processing. Defaults to number of CPU cores.',
        default=os.cpu_count(), type=int)
    parser.add_argument('--chunksize', help='Chunk size to be used in multiprocessing mapping.',
        default=1, type=int)
    parser.add_argument('--exclude', help='Exclude path prefixes in given file.', type=str,
        default=None)
    parser.add_argument('--blame-C', help="Git blame -C option. To not use -C provide ''", type=str,
        dest='blame_C', default='-C')

    args = parser.parse_args()

    mine_git_repo(args.repo, args.outfile, use_blocks=args.use_blocks,
                  no_of_processes=args.numprocesses, chunksize=args.chunksize, exclude=args.exclude,
                  blame_C=args.blame_C)
