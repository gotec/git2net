#!/usr/bin/python3

#################################################
# All functions that work on the git repository #
#################################################

import sqlite3
import os
import argparse

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
import re
import lizard


def _get_block_length(lines, k):
    """ Calculates the length (in number of lines) of a edit of added/deleted lines starting in a
        given line k.

    Args:
        lines: dictionary of added or deleted lines
        k: line number to check for

    Returns:
        block_size: number of lines in the contiguously block that was modified
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


def _identify_edits(deleted_lines, added_lines, use_blocks=False):
    """ Maps line numbers between the pre- and post-commit version of a modification.

    Args:
        deleted_lines: dictionary of deleted lines
        added_lines: dictionary of added lines
        use_blocks: bool indicating whether or not to use the block approach

    Returns:
        pre_to_post: dictionary mapping line numbers before and after the commit
        edits: dataframe with information on edits
    """

    # either deleted or added lines must contain items otherwise there would not be a modification
    # to process
    if len(deleted_lines) > 0:
        max_deleted = max(deleted_lines.keys())
        min_deleted = min(deleted_lines.keys())
    else:
        max_deleted = -1
        min_deleted = np.inf

    if len(added_lines) > 0:
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
            length_added_block = _get_block_length(added_lines, post)
            length_deleted_block = _get_block_length(deleted_lines, pre)

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
    """ Computes entropy for a given text based on UTF8 alphabet.

    Args:
        text: string to compute the text entropy for

    Returns:
        text_entropy: text entropy of the given string
    """

    if len(text) == 0:
        text_entropy = None
    else:
        text_entropy = entropy([text.count(chr(i)) for i in range(256)], base=2)

    return text_entropy


def get_commit_dag(repo_string):
    """ Extracts commit dag from given path to git repository.

    Args:
        repo_string: path to the git repository that is mined

    Returns:
        dag: dag linking successive commits in the same branch
    """
    git_repo = pydriller.GitRepository(repo_string)
    commits = [x.hash[0:7] for x in git_repo.get_list_commits()]
    dag = pp.DAG()
    for node in commits:
        for parent in git_repo.get_commit(node).parents:
            dag.add_edge(parent[0:7], node)
    return dag


def _parse_blame_C(blame_C):
    """ Converts the input provided for the copy option in git blame to a list of options required
        as input for gitpython.

    Args:
        blame_C: string defining how the copy option in git blame is used

    Returns:
        list_of_arguments: list of parameters for gitpython blame
    """
    pattern = re.compile("(^$|^-?C{0,3}[0-9]*$)")
    if not pattern.match(blame_C):
        raise Exception("Invalid 'blame_C' supplied.")
    if len(blame_C) == 0:
        list_of_arguments = []
    else:
        if blame_C[0] == '-':
            blame_C = blame_C[1:]
        cs = len(blame_C) - len(blame_C.lstrip('C'))
        num = blame_C.lstrip('C')
        list_of_arguments = ['-C' for i in range(cs - 1)] + ['-C{}'.format(num)]
    return list_of_arguments


def _parse_porcelain_blame(blame):
    """ Parses the porcelain output of git blame and returns content as dataframe.

    Args:
        blame: porcelain output of git blame

    Returns:
        blame_info: content of blame as pandas dataframe
    """
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
    blame_info = pd.DataFrame(l)
    return blame_info


def _get_edit_details(edit, commit, deleted_lines, added_lines, blame_info_parent,
                      blame_info_commit, use_blocks=False):
    """ Extracts detailed measures for a given edit.

    Args:
        edit: edit as identified in _identify_edits
        commit: pydriller commit object containing the edit
        deleted_lines: dict of added lines
        added_lines: dict of deleted lines
        blame_info_parent: blame info for parent commit as output from _parse_porcelain_blame
        blame_info_commit: blame info for current commit as output from _parse_porcelain_blame
        use_blocks: bool indicating whether or not to use the block approach

    Returns:
        e: pandas dataframe containing information on edits
    """
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
        raise Exception("Unexpected error in '_get_edit_details'.")

    return e

def _extract_edits(git_repo, commit, modification, use_blocks=False, blame_C='-C'):
    """ Returns dataframe with metadata on edits made in a given modification.

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
        _, edits = _identify_edits(deleted_lines, added_lines, use_blocks=use_blocks)

    # In order to trace the origins of lines e execute git blame is executed. For lines that were
    # deleted with the current commit, the blame needs to be executed on the parent commit. As
    # merges are treated separately, commits should only have one parent. For added lines, git blame
    # is executed on the current commit.
    blame_info_parent = None
    blame_info_commit = None
    if len(deleted_lines) > 0:
        assert len(commit.parents) == 1
        blame_parent = git_repo.git.blame(commit.parents[0],
                                          _parse_blame_C(blame_C) +
                                          ['-w', '--show-number', '--porcelain'],
                                          modification.old_path)
        blame_info_parent = _parse_porcelain_blame(blame_parent)

    if len(added_lines) > 0:
        blame_commit = git_repo.git.blame(commit.hash,
                                          _parse_blame_C(blame_C) +
                                          ['-w', '--show-number', '--porcelain'],
                                          modification.new_path)
        blame_info_commit = _parse_porcelain_blame(blame_commit)

    # Next, metadata on all identified edits is extracted and added to a pandas DataFrame.
    edits_info = pd.DataFrame()
    for _, edit in edits.iterrows(): #tqdm(edits.iterrows(), leave=False, desc='edits ' + commit.hash[0:7] + ' ' + str(modification.filename), total=len(edits)):
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

        e.update(_get_edit_details(edit, commit, deleted_lines, added_lines, blame_info_parent,
                                  blame_info_commit, use_blocks=use_blocks))

        edits_info = edits_info.append(e, ignore_index=True, sort=False)

    return edits_info

def _extract_edits_merge(git_repo, commit, modification_info, use_blocks=False, blame_C='-C'):
    """ Returns dataframe with metadata on edits made in a given modification for merge commits.

    Args:
        git_repo: pydriller GitRepository object
        commit: pydriller Commit object
        modification: pydriller Modification object
        use_blocks: bool, determins if analysis is performed on block or line basis
        blame_C: git blame '-C' option. By default, '-C' is used.

    Returns:
        edits_info: pandas DataFrame object containing metadata on all edits in given modification
    """
    assert commit.merge
    # With merges, the following cases can occur:
    #   1. Changes of one or more parents are accepted.
    #   2. Changes made prior to the merge are replaced with new edits.
    # To obtain the state of the file before merging, get blame is executed on all parent commits.
    parent_blames = []
    for parent in commit.parents:
        blame = git_repo.git.blame(parent,
                                   _parse_blame_C(blame_C) +
                                   ['-w', '--show-number', '--porcelain'],
                                   modification_info['old_path'])
        parent_blame = _parse_porcelain_blame(blame).rename(
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
                               _parse_blame_C(blame_C) +
                               ['-w', '--show-number', '--porcelain'],
                               modification_info['new_path'])
    current_blame = _parse_porcelain_blame(blame).rename(
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

        _, edits = _identify_edits(deleted_lines, added_lines, use_blocks=use_blocks)

        for _, edit in edits.iterrows(): #tqdm(edits.iterrows(), leave=False, desc='edits ' + commit.hash[0:7] + ' ' + str(modification_info['filename']), total=len(edits)):
            e = {}
            # Extract general information.
            e['commit_hash'] = commit.hash
            e['edit_type'] = edit.type
            e.update(modification_info)
            e.update(_get_edit_details(edit, commit, deleted_lines, added_lines, parent_blames[idx],
                                    current_blame, use_blocks=use_blocks))

            edits_info = edits_info.append(e, ignore_index=True, sort=False)

    return edits_info


def _get_edited_file_paths_since_split(git_repo, commit):
    """ For a merge commit returns list of all files edited since the last creation of a new branch
        relevant for the merge.

    Args:
        git_repo: pydriller GitRepository object
        commit: pydriller Commit object

    Returns:
        edited_file_paths: list of paths to the edited files
    """
    def expand_dag(dag, leafs):
        """ Expands a dag by adding the parents of a given set of nodes to the dag.

        Args:
            dag: pathpy DAG object
            leafs: set of nodes that are expanded

        Returns:
            dag: the expanded pathpy DAG object
        """
        for node in leafs:
            parents = git_repo.get_commit(node).parents
            for parent in parents:
                dag.add_edge(node, parent)
        return dag
    def common_node_on_paths(paths):
        """ Computes the overlap between given sets of nodes. Returns the nodes present in all sets.

        Args:
            paths: list of node sequences

        Returns:
            common_nodes: set of nodes that are present on all paths
        """
        # Drop first and last element of the path.
        common_nodes = set(paths[0][1:-1])
        for path in paths[1:]:
            common_nodes.intersection_update(path[1:-1])
        common_nodes = list(common_nodes)
        return common_nodes

    def remove_successors(dag, node):
        """ Removes all successors of a node from a given dag.

        Args:
            dag: pathpy DAG object
            node: node for which successors shall be removed

        Returns:
            dag: reduced pathpy DAG object
        """
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

    for node in common_nodes:
        dag = remove_successors(dag, node)

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


def _process_commit(args):
    """ Extracts information on commit and all edits made with the commit.

    Args:
        args: dictionary with arguments. For multiprocessing, function can only take single input.
              Dictionary must contain:
                  repo_string: path to the git repository that is mined
                  commit_hash: hash of the commit that is processed
                  use_blocks: bool, determins if analysis is performed on block or line basis
                  exclude_paths: file paths that are excluded from the analysis
                  blame_C: string for the blame C option

    Returns:
        extracted_result: dict containing two dataframes with information of commit and edits
    """
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
        edited_file_paths = _get_edited_file_paths_since_split(git_repo, commit)
        for edited_file_path in edited_file_paths:
            exclude_file = False
            for x in args['exclude_paths']:
                if edited_file_path.startswith(x + os.sep) or (edited_file_path == x):
                    exclude_file = True
            if not exclude_file:
                modification_info = {}
                try:
                    file_contents = git_repo.git.show('{}:{}'.format(commit.hash, edited_file_path))
                    l = lizard.analyze_file.analyze_source_code(edited_file_path, file_contents)

                    modification_info['filename'] = edited_file_path.split(os.sep)[-1]
                    modification_info['new_path'] = edited_file_path
                    modification_info['old_path'] = edited_file_path
                    modification_info['cyclomatic_complexity_of_file'] = l.CCN
                    modification_info['lines_of_code_in_file'] = l.nloc
                    modification_info['modification_type'] = 'merge_self_accept'

                    df_edits = df_edits.append(_extract_edits_merge(git_repo, commit,
                                                                   modification_info,
                                                                   use_blocks=args['use_blocks'],
                                                                   blame_C=args['blame_C']),
                                               ignore_index=True, sort=True)
                except GitCommandError:
                    # A GitCommandError occurs if the file was deleted. In this case it currently
                    # has no content.

                    # Get filenames from all modifications in merge commit.
                    paths = [m.old_path for m in commit.modifications]

                    # Analyse changes if modification was recorded. Else, the deletions were made
                    # before the merge.
                    if edited_file_path in paths:
                        modification_info['filename'] = edited_file_path.split(os.sep)[-1]
                        modification_info['new_path'] = None # File was deleted.
                        modification_info['old_path'] = edited_file_path
                        modification_info['cyclomatic_complexity_of_file'] = 0
                        modification_info['lines_of_code_in_file'] = 0
                        modification_info['modification_type'] = 'merge_self_accept'

                        df_edits = df_edits.append(_extract_edits_merge(git_repo, commit,
                                                                    modification_info,
                                                                    use_blocks=args['use_blocks'],
                                                                    blame_C=args['blame_C']),
                                                   ignore_index=True, sort=True)

    else:
        for modification in commit.modifications:#tqdm(commit.modifications, leave=False, desc='modifications ' + commit.hash[0:7]):
            exclude_file = False
            for x in args['exclude_paths']:
                if modification.new_path:
                    if modification.new_path.startswith(x + os.sep) or (modification.new_path == x):
                        exclude_file = True
                if not exclude_file and modification.old_path:
                    if modification.old_path.startswith(x + os.sep):
                        exclude_file = True
            if not exclude_file:
                df_edits = df_edits.append(_extract_edits(git_repo, commit, modification,
                                                         use_blocks=args['use_blocks'],
                                                         blame_C=args['blame_C']),
                                           ignore_index=True, sort=True)


    df_commit = pd.DataFrame(c, index=[0])

    extracted_result = {'commit': df_commit, 'edits': df_edits}

    return extracted_result


def _process_repo_serial(repo_string, sqlite_db_file, use_blocks=False, exclude=None, blame_C='-C',
                        _p_commits=[]):
    """ Processes all commits in a given git repository in a serial manner.

    Args:
        repo_string: path to the git repository that is mined
        sqlite_db_file: path (including database name) where the sqlite database will be created
        use_blocks: bool, determins if analysis is performed on block or line basis
        exclude: file paths that are excluded from the analysis
        blame_C: string for the blame C option
        _p_commits: list of commits that are already in the database

    Returns:
        sqlite database will be written at specified location
    """

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
        result = _process_commit(args)

        if not result['commit'].empty:
            result['commit'].to_sql('commits', con, if_exists='append')
        if not result['edits'].empty:
            result['edits'].to_sql('edits', con, if_exists='append')


def _process_repo_parallel(repo_string, sqlite_db_file, use_blocks=False,
                          no_of_processes=os.cpu_count(), chunksize=1, exclude=None, blame_C='-C',
                          _p_commits=[]):
    """ Processes all commits in a given git repository in a parallel manner.

    Args:
        repo_string: path to the git repository that is mined
        sqlite_db_file: path (including database name) where the sqlite database will be created
        use_blocks: bool, determins if analysis is performed on block or line basis
        no_of_processes: number of parallel processes that are spawned
        chunksize: number of tasks that are assigned to a process at a time
        exclude: file paths that are excluded from the analysis
        blame_C: string for the blame C option
        _p_commits: list of commits that are already in the database

    Returns:
        sqlite database will be written at specified location
    """
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
        for result in p.imap_unordered(_process_commit, args, chunksize=chunksize):
            if not result['commit'].empty:
                result['commit'].to_sql('commits', con, if_exists='append')
            if not result['edits'].empty:
                result['edits'].to_sql('edits', con, if_exists='append')
            pbar.update(1)


def identify_file_renaming(repo_string):
    """ Identifies all names and locations different files in a repository have had.

    Args:
        repo_string: path to the git repository that is mined

    Returns:
        dag: pathpy DAG object depicting the renaming process
        aliases: dictionary containing all aliases for all files
    """

    # TODO: Consider corner case where file is renamed and new file with old name is created.
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

    def _get_path_to_leaf_node(dag, node, _path=[]):
        """ For a given node and dag returns path to leaf node with leaf.

        Args:
            dag: pathpy DAG object
            node: node for which path is computed
            _path=[]: subpath used for recursion

        Returns:
            path: path from the current node to a leaf node in the dag

        """
        if len(dag.successors[node]) > 0:
            return _get_path_to_leaf_node(dag, list(dag.successors[node])[0], _path=[node] + _path)
        else:
            path = [node] + _path
            return path

    renamings = []
    for node in dag.nodes:
        if 'added file' in dag.predecessors[node]:
            renamings.append(_get_path_to_leaf_node(dag, node))

    aliases = {}
    for renaming in renamings:
        if 'deleted file' in renaming:
            renaming.remove('deleted file')
        for alias in renaming:
            aliases[alias] = renaming

    return dag, aliases


def get_unified_changes(repo_string, commit_hash, file_path):
    """ Returns dataframe with github-like unified diff representation of the content of a file
        before and after a commit for a given git repository, commit hash and file path.

    Args:
        repo_string: path to the git repository that is mined
        commit_hash: commit hash for which the changes are computed
        file_path: path to file (within the repository) for which the changes are computed

    Returns:
        df: pandas dataframe listing changes made to file in commit
    """
    git_repo = pydriller.GitRepository(repo_string)
    commit = git_repo.get_commit(commit_hash)

    # Select the correct modifictaion.
    for modification in commit.modifications:
        if modification.new_path == file_path:
            break

    # Parse the diff extracting the lines added and deleted with the given commit.
    parsed_lines = git_repo.parse_diff(modification.diff)

    deleted_lines = { x[0]:x[1] for x in parsed_lines['deleted'] }
    added_lines = { x[0]:x[1] for x in parsed_lines['added'] }

    # Indetify the edits made with the changes.
    pre_to_post, edits = _identify_edits(deleted_lines, added_lines)

    # Extract the source code after the commit.
    post_source_code = modification.source_code.split('\n')

    # Initialise lists for output.
    pre_line_no = []
    post_line_no = []
    action = []
    code = []

    # Go through all lines and report on the changes.
    pre_counter = 1
    post_counter = 1
    while post_counter < len(post_source_code) or \
          pre_counter < max(deleted_lines.keys()) or \
          post_counter < max(added_lines.keys()):
        if pre_counter in list(edits.pre_start):
            pre_line_no.append(pre_counter)
            post_line_no.append(None)
            action.append('-')
            code.append(deleted_lines[pre_counter])
            pre_counter += 1
        elif post_counter in list(edits.post_start):
            pre_line_no.append(None)
            post_line_no.append(post_counter)
            action.append('+')
            code.append(added_lines[post_counter])
            post_counter += 1
        else:
            pre_line_no.append(pre_counter)
            post_line_no.append(post_counter)
            action.append(None)
            code.append(post_source_code[post_counter - 1])
            pre_counter += 1
            post_counter += 1

    df = pd.DataFrame({'pre': pre_line_no, 'post': post_line_no, 'action': action, 'code': code})

    return df

def mine_git_repo(repo_string, sqlite_db_file, use_blocks=False,
                  no_of_processes=os.cpu_count(), chunksize=1, exclude=[], blame_C='-C'):
    """ Creates sqlite database with details on commits and edits for a given git repository.

    Args:
        repo_string: path to the git repository that is mined
        sqlite_db_file: path (including database name) where the sqlite database will be created
        use_blocks: bool, determins if analysis is performed on block or line basis
        no_of_processes: number of parallel processes that are spawned
        chunksize: number of tasks that are assigned to a process at a time
        exclude: file paths that are excluded from the analysis
        blame_C: string for the blame C option

    Returns:
        sqlite database will be written at specified location
    """

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
        _process_repo_parallel(repo_string=repo_string, sqlite_db_file=sqlite_db_file,
                              use_blocks=use_blocks, no_of_processes=no_of_processes,
                              chunksize=chunksize, exclude=exclude, blame_C=blame_C,
                              _p_commits=p_commits)
    else:
        _process_repo_serial(repo_string=repo_string, sqlite_db_file=sqlite_db_file,
                            use_blocks=use_blocks, exclude=exclude, blame_C=blame_C,
                            _p_commits=p_commits)

####################################################################################################

### Allows to use git2net from the command line
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
