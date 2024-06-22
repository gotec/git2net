#!/usr/bin/python3

#################################################
# All functions that work on the git repository #
#################################################

import sqlite3
import os
from subprocess import check_output

import multiprocessing

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
from scipy.stats import entropy

import pydriller as pydriller
from Levenshtein import distance as lev_dist
import datetime

import pathpy as pp
import re
import collections
import git
from git.exc import GitCommandError

from git2net import __version__

import json

import contextlib
import io

import threading
import ctypes

import logging

git_init_lock = multiprocessing.Lock()


abs_path = os.path.dirname(__file__)
rel_path = 'helpers/binary-extensions/binary-extensions.json'
with open(os.path.join(abs_path, rel_path)) as json_file:
    binary_extensions = json.load(json_file)


class TimeoutException(Exception):
    pass


class Timeout():
    """
    Context manager that raises TimeoutException after wait of length timeout.
    If timeout is <= 0, no timer is started.

    """
    def __init__(self, timeout):
        self.timeout = timeout
        self.timed_out = False
        self.active = False
        self.target_tid = threading.current_thread().ident
        self.timer = None

    def __enter__(self):
        if self.timeout > 0:
            self.timer = threading.Timer(self.timeout, self.stop)
            self.timer.start()
            self.active = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.active:
            self.timer.cancel()
            self.active = False
        if exc_type == TimeoutException:
            return True
        return False

    def stop(self):
        self.timed_out = True
        # raise TimeoutException in main thread. Inspired by implementation in 'stopit'
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(self.target_tid),
            ctypes.py_object(TimeoutException)
        )


def _get_block_length(lines, k):
    """
    Calculates the length (in number of lines) of a edit of added/deleted
    lines starting in a given line k.

    :param dict lines: dictionary of added or deleted lines
    :param int k: line number to check for

    :return:
        *int* – number of lines in the contiguously block that was modified
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


def _identify_edits(deleted_lines, added_lines, extraction_settings):
    """
    Maps line numbers between the pre- and post-commit version of a
    modification.

    :param dict deleted_lines: dictionary of deleted lines
    :param dict added_lines: dictionary of added lines
    :param dict extraction_settings: settings for the extraction

    :return:
        - *dict* – dictionary mapping line numbers before and after commit
        - *pandas.DataFrame* – dataframe with information on edits
    """

    # either deleted or added lines must contain items otherwise there would
    # not be a modification to process
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
    edits = []

    # line numbers of lines before the first addition or deletion do not change
    pre = min(max(min_added, 0), max(min_deleted, 0))
    post = min(max(min_added, 0), max(min_deleted, 0))

    # counters used to match pre and post line number
    no_post_inc = 0
    both_inc = 0
    no_pre_inc = 0

    # line numbers after the last addition or deletion do not matter for edits
    while (pre <= max_deleted + 1) or (post <= max_added + 1):
        if extraction_settings['use_blocks']:
            # compute size of added and deleted edits
            # size is reported as 0 if the line is not in added or deleted
            # lines, respectively
            length_added_block = _get_block_length(added_lines, post)
            length_deleted_block = _get_block_length(deleted_lines, pre)

            # replacement if both deleted and added > 0
            # if not both > 0, deletion if deleted > 0
            # if not both > 0, addition if added > 0
            if (length_deleted_block > 0) and (length_added_block > 0):
                edits.append({'pre_start': int(pre),
                              'number_of_deleted_lines':
                                  int(length_deleted_block),
                              'post_start': int(post),
                              'number_of_added_lines': int(length_added_block),
                              'type': 'replacement'})
            elif length_deleted_block > 0:
                edits.append({'pre_start': int(pre),
                              'num_start': int(post),
                              'number_of_deleted_lines':
                                  int(length_deleted_block),
                              'poster_of_added_lines': int(length_added_block),
                              'type': 'deletion'})
            elif length_added_block > 0:
                edits.append({'pre_start': int(pre),
                              'number_of_deleted_lines':
                                  int(length_deleted_block),
                              'post_start': int(post),
                              'number_of_added_lines': int(length_added_block),
                              'type': 'addition'})

            # deleted edit is larger than added edit
            if length_deleted_block > length_added_block:
                no_post_inc = length_deleted_block - length_added_block
                both_inc = length_added_block
            # added edit is larger than deleted edit
            elif length_added_block > length_deleted_block:
                no_pre_inc = length_added_block - length_deleted_block
                both_inc = length_deleted_block
        else:  # no blocks are considered
            pre_in_deleted = pre in deleted_lines
            post_in_added = post in added_lines
            # cf. case of blocks above
            # length of blocks is equivalent to line being in added or deleted
            # lines
            if pre_in_deleted and post_in_added:
                edits.append({'pre_start': int(pre),
                              'number_of_deleted_lines': int(pre_in_deleted),
                              'post_start': int(post),
                              'number_of_added_lines': int(post_in_added),
                              'type': 'replacement'})
            elif pre_in_deleted and not post_in_added:
                edits.append({'pre_start': int(pre),
                              'number_of_deleted_lines': int(pre_in_deleted),
                              'post_start': None,
                              'number_of_added_lines': None,
                              'type': 'deletion'})
                no_post_inc += 1
            elif post_in_added and not pre_in_deleted:
                edits.append({'pre_start': None,
                              'number_of_deleted_lines': None,
                              'post_start': int(post),
                              'number_of_added_lines': int(post_in_added),
                              'type': 'addition'})
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

    edits = pd.DataFrame(edits)
    return pre_to_post, edits


def text_entropy(text):
    """
    Computes entropy for a given text based on UTF8 alphabet.

    :param str text: string to compute the text entropy for

    :return:
        *float* – text entropy of the given string
    """
    # we only consider UTF8 characters to compute the text entropy
    pk = [text.count(chr(i)) for i in range(256)]
    if sum(pk) == 0:
        text_entropy = None
    else:
        text_entropy = entropy(pk, base=2)
    return text_entropy


def get_commit_dag(git_repo_dir):
    """
    Extracts commit dag from given path to git repository.

    :param str git_repo_dir: path to the git repository that is mined

    :return:
        *pathpy.DAG* – dag linking successive commits in the same branch
    """
    git_repo = pydriller.Git(git_repo_dir)
    commits = [x.hash[0:7] for x in git_repo.get_list_commits()]
    dag = pp.DAG()
    for node in commits:
        for parent in git_repo.get_commit(node).parents:
            dag.add_edge(parent[0:7], node)
    return dag


def _parse_blame_C(blame_C):
    """
    Converts the input provided for the copy option in git blame to a list
    of options required as input for gitpython.

    :param str blame_C: string defining how the copy option in git blame is used

    :return:
        *list* – list of parameters for gitpython blame
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
        list_of_arguments = ['-C' for i in range(cs - 1)] + \
                            ['-C{}'.format(num)]
    return list_of_arguments


def _parse_porcelain_blame(blame):
    """
    Parses the porcelain output of git blame and returns content as
    dataframe.

    :param str blame: porcelain output of git blame

    :return:
        *dict* – content of blame as pandas dataframe
    """
    line_dict = {
        'original_commit_hash': [],
        'original_line_no': [],
        'original_file_path': [],
        'line_content': [],
        'line_number': []
    }
    start_of_line_info = True
    prefix = '\t'
    line_number = 1
    filename = ''  # Initialise filename variable.
    for idx, line in enumerate(blame.split('\n')):
        if line.startswith(prefix):
            line_dict['original_file_path'].append(filename)
            line_dict['line_content'].append(line[len(prefix):])
            line_dict['line_number'].append(line_number)
            line_number += 1
            start_of_line_info = True
        else:
            entries = line.split(' ')
            if start_of_line_info:
                line_dict['original_commit_hash'].append(entries[0])
                line_dict['original_line_no'].append(entries[1])
                start_of_line_info = False
            elif entries[0] == 'filename':
                filename = entries[1]

    blame_info = pd.DataFrame(line_dict)
    return blame_info


def _get_edit_details(edit, commit, deleted_lines, added_lines,
                      blame_info_parent, blame_info_commit,
                      extraction_settings):
    """
    Extracts detailed measures for a given edit.

    :param dict edit: edit as identified in _identify_edits
    :param pydriller.Commit commit: pydriller commit object containing the edit
    :param dict deleted_lines: dict of added lines
    :param dict added_lines: dict of deleted lines
    :param str blame_info_parent: blame info for parent commit as output from _parse_porcelain_blame
    :param str blame_info_commit: blame info for current commit as output from _parse_porcelain_blame
    :param dict extraction_settings: settings for the extraction

    :return:
        *pandas.DataFrame* – pandas dataframe containing information on edits
    """
    # Different actions for different types of edits.
    e = {}
    if edit.type == 'replacement':
        # For replacements, both the content of the deleted and added block are
        # required in order to compute text entropy, as well as Levenshtein
        # edit distance between them.
        deleted_block = []
        for i in range(int(edit.pre_start),
                       int(edit.pre_start + edit.number_of_deleted_lines)):
            deleted_block.append(deleted_lines[i])

        added_block = []
        for i in range(int(edit.post_start),
                       int(edit.post_start + edit.number_of_added_lines)):
            added_block.append(added_lines[i])

        # For the analysis, lines are concatenated with whitespaces.
        deleted_block = ' '.join(deleted_block)
        added_block = ' '.join(added_block)

        # Given this, all metadata can be written.
        # Data on the content and location of deleted line in the parent
        # commit.
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
        if extraction_settings['extract_text']:
            e['pre_text'] = deleted_block.encode('utf8', 'surrogateescape') \
                                         .decode('utf8', 'replace')
            e['post_text'] = added_block.encode('utf8', 'surrogateescape') \
                                        .decode('utf8', 'replace')
        e['levenshtein_dist'] = lev_dist(deleted_block, added_block)

        # Data on origin of deleted line. Every deleted line must have an
        # origin.
        if extraction_settings['use_blocks']:
            e['original_commit_deletion'] = 'not available with use_blocks'
            e['original_line_no_deletion'] = 'not available with use_blocks'
            e['original_file_path_deletion'] = 'not available with use_blocks'
        else:
            assert blame_info_parent is not None
            e['original_commit_deletion'] = blame_info_parent.at[
                                                int(edit.pre_start) - 1,
                                                'original_commit_hash']
            e['original_line_no_deletion'] = blame_info_parent.at[
                                                 int(edit.pre_start) - 1,
                                                 'original_line_no']
            e['original_file_path_deletion'] = blame_info_parent.at[
                                                   int(edit.pre_start) - 1,
                                                   'original_file_path']

        # Data on the origin of added line. Can be either original or copied
        # form other file.
        if extraction_settings['use_blocks']:
            e['original_commit_addition'] = 'not available with use_blocks'
            e['original_line_no_addition'] = 'not available with use_blocks'
            e['original_file_path_addition'] = 'not available with use_blocks'
        elif blame_info_commit.at[int(edit.post_start) - 1,
                                  'original_commit_hash'] == commit.hash:
            # The line is original, there exists no original commit, line
            # number or file path.
            e['original_commit_addition'] = None
            e['original_line_no_addition'] = None
            e['original_file_path_addition'] = None
        else:
            # The line was copied from somewhere.
            assert blame_info_commit is not None
            e['original_commit_addition'] = blame_info_commit.at[
                                                int(edit.post_start) - 1,
                                                'original_commit_hash']
            e['original_line_no_addition'] = blame_info_commit.at[
                                                 int(edit.post_start) - 1,
                                                 'original_line_no']
            e['original_file_path_addition'] = blame_info_commit.at[
                                                   int(edit.post_start) - 1,
                                                   'original_file_path']

    elif edit.type == 'deletion':
        # For deletions, only the content of the deleted block is required.
        deleted_block = []
        for i in range(int(edit.pre_start),
                       int(edit.pre_start + edit.number_of_deleted_lines)):
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

        # Levenshtein edit distance is set to 'None'. Theoretically one
        # keystroke is required.
        if extraction_settings['extract_text']:
            e['pre_text'] = deleted_block.encode('utf8', 'surrogateescape') \
                                         .decode('utf8', 'replace')
            e['post_text'] = None
        e['levenshtein_dist'] = len(deleted_block)

        # Data on origin of deleted line. Every deleted line must have a
        #  origin.
        if extraction_settings['use_blocks']:
            e['original_commit_deletion'] = 'not available with use_blocks'
            e['original_line_no_deletion'] = 'not available with use_blocks'
            e['original_file_path_deletion'] = 'not available with use_blocks'
        else:
            assert blame_info_parent is not None
            e['original_commit_deletion'] = blame_info_parent.at[
                                                int(edit.pre_start) - 1,
                                                'original_commit_hash']
            e['original_line_no_deletion'] = blame_info_parent.at[
                                                 int(edit.pre_start) - 1,
                                                 'original_line_no']
            e['original_file_path_deletion'] = blame_info_parent.at[
                                                   int(edit.pre_start) - 1,
                                                   'original_file_path']

    elif edit.type == 'addition':
        # For additions, only the content of the added block is required.
        added_block = []
        for i in range(int(edit.post_start),
                       int(edit.post_start + edit.number_of_added_lines)):
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

        # Levenshtein edit distance is length of added block as nothing existed
        # before.
        if extraction_settings['extract_text']:
            e['pre_text'] = None
            e['post_text'] = added_block.encode('utf8', 'surrogateescape') \
                                        .decode('utf8', 'replace')
        e['levenshtein_dist'] = len(added_block)

        # If the lines were newly added to this file, they might still come
        # from another file.
        if extraction_settings['use_blocks']:
            e['original_commit_addition'] = 'not available with use_blocks'
            e['original_line_no_addition'] = 'not available with use_blocks'
            e['original_file_path_addition'] = 'not available with use_blocks'
        elif blame_info_commit.at[int(edit.post_start) - 1,
                                  'original_commit_hash'] == commit.hash:
            # The line is original, there exists no original commit, line
            # number or file path.
            e['original_commit_addition'] = None
            e['original_line_no_addition'] = None
            e['original_file_path_addition'] = None
        else:
            # The line was copied from somewhere.
            assert blame_info_commit is not None
            e['original_commit_addition'] = blame_info_commit.at[
                                                int(edit.post_start) - 1,
                                                'original_commit_hash']
            e['original_line_no_addition'] = blame_info_commit.at[
                                                 int(edit.post_start) - 1,
                                                 'original_line_no']
            e['original_file_path_addition'] = blame_info_commit.at[
                                                   int(edit.post_start) - 1,
                                                   'original_file_path']

    elif (edit.type == 'file_renaming') or (edit.type == 'binary_file_change'):
        # For file renaming only old and new path are required which were
        # already set before.
        e['pre_starting_line_no'] = None
        e['pre_len_in_lines'] = None
        e['pre_len_in_chars'] = None
        e['pre_entropy'] = None
        e['post_starting_line_no'] = None
        e['post_len_in_lines'] = None
        e['post_len_in_chars'] = None
        e['post_entropy'] = None

        if extraction_settings['use_blocks']:
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
        if extraction_settings['extract_text']:
            e['pre_text'] = None
            e['post_text'] = None
        e['levenshtein_dist'] = 0

    else:
        LOG = logging.getLogger('git2net')
        LOG.error(edit.type)
        raise Exception("Unexpected error in '_get_edit_details'.")

    return e


def is_binary_file(filename, file_content):
    """
    Detects if a file with given content is a binary file.

    :param str filename: name of the file including its file extension
    :param str file_content: content of the file

    :returns:
        *bool* – True if binary file is detected, otherwise False
    """

    if filename is None:
        return False
    else:
        try:
            extension = re.search(r'.*\.([^\.]+)$', filename).groups()[0]
        except AttributeError:
            extension = None

        if extension in binary_extensions:
            return True
        else:
            try:
                file_content.encode('utf-8', errors='strict')
            except UnicodeEncodeError:
                return True
            else:
                return False


def _extract_edits(git_repo, commit, modification, extraction_settings):
    """
    Returns dataframe with metadata on edits made in a given modification.

    :param pydriller.Git git_repo: pydriller Git object
    :param pydriller.Commit commit: pydriller Commit object
    :param pydriller.ModifiedFile modification: pydriller ModifiedFile object
    :param dict extraction_settings: settings for the extraction

    :return:
        *pandas.DataFrame* – pandas DataFrame object containing metadata on all edits in given modification
    """

    binary_file = is_binary_file(modification.filename, modification.diff)
    found_paths = False

    if not binary_file:
        try:
            old_path, new_path = \
                re.search(r'Binary files a?\/(.*) and b?\/(.*) differ',
                          modification.diff.strip()).groups()

            if old_path == 'dev/null':
                old_path = None
            if new_path == 'dev/null':
                new_path = None

            found_paths = True
            binary_file = True
        except AttributeError:
            pass

    if binary_file:
        if found_paths:
            edits = pd.DataFrame({'pre_start': None,
                                  'number_of_deleted_lines': None,
                                  'post_start': None,
                                  'number_of_added_lines': None,
                                  'type': 'binary_file_change',
                                  'new_path': new_path,
                                  'old_path': old_path}, index=[0])
        else:
            edits = pd.DataFrame({'pre_start': None,
                                  'number_of_deleted_lines': None,
                                  'post_start': None,
                                  'number_of_added_lines': None,
                                  'type': 'binary_file_change',
                                  'new_path': modification.new_path,
                                  'old_path': modification.old_path},
                                 index=[0])
        deleted_lines = {}
        added_lines = {}
    else:
        # Parse diff of given modification to extract added and deleted lines
        parsed_lines = modification.diff_parsed

        deleted_lines = {x[0]: x[1] for x in parsed_lines['deleted']}
        added_lines = {x[0]: x[1] for x in parsed_lines['added']}

        # If there was a modification but no lines were added or removed, the
        # file was renamed.
        if (len(deleted_lines) == 0) and (len(added_lines) == 0):
            edits = pd.DataFrame({'pre_start': None,
                                  'number_of_deleted_lines': None,
                                  'post_start': None,
                                  'number_of_added_lines': None,
                                  'type': 'file_renaming'}, index=[0])
        else:
            # If there were lines added or deleted, the specific edits are
            # identified.
            _, edits = _identify_edits(deleted_lines, added_lines,
                                       extraction_settings)

    # In order to trace the origins of lines e execute git blame is executed.
    # For lines that were  deleted with the current commit, the blame needs to
    # be executed on the parent commit. As merges are treated separately,
    # commits should only have one parent. For added lines, git blame is
    # executed on the current commit.
    blame_info_parent = None
    blame_info_commit = None

    try:
        if not binary_file:
            if len(deleted_lines) > 0:
                assert len(commit.parents) == 1
                blame_parent = git.Git(str(git_repo.path)) \
                                  .blame(commit.parents[0],
                                         extraction_settings['blame_options'],
                                         modification.old_path)
                blame_info_parent = _parse_porcelain_blame(blame_parent)

            if len(added_lines) > 0:
                blame_commit = git.Git(str(git_repo.path)) \
                                  .blame(commit.hash,
                                         extraction_settings['blame_options'],
                                         modification.new_path)
                blame_info_commit = _parse_porcelain_blame(blame_commit)

    except GitCommandError:
        return pd.DataFrame()
    else:
        # Next, metadata on all identified edits is extracted and added to a
        # pandas DataFrame.
        line_dict = []
        for _, edit in edits.iterrows():
            e = {}
            # Extract general information.
            if edit.type == 'binary_file_change':
                e['new_path'] = edit.new_path
                e['old_path'] = edit.old_path
                e['total_added_lines'] = None
                e['total_removed_lines'] = None
            else:
                e['new_path'] = modification.new_path
                e['old_path'] = modification.old_path
                e['total_added_lines'] = modification.added_lines
                e['total_removed_lines'] = modification.deleted_lines
            e['filename'] = modification.filename
            e['commit_hash'] = commit.hash
            e['modification_type'] = modification.change_type.name
            e['edit_type'] = edit.type

            e.update(_get_edit_details(edit, commit, deleted_lines,
                                       added_lines, blame_info_parent,
                                       blame_info_commit, extraction_settings))

            line_dict.append(e)

        edits_info = pd.DataFrame(line_dict)
        return edits_info


def _extract_edits_merge(git_repo, commit, modification_info,
                         extraction_settings):
    """
    Returns dataframe with metadata on edits made in a given modification
    for merge commits.

    :param str git_repo: pydriller Git object
    :param pydriller.Commit commit: pydriller Commit object
    :param pydriller.ModifiedFile modification_info: information on the modification as stored in a
        pydriller ModifiedFile.
    :param dict extraction_settings: settings for the extraction

    :return:
        *pandas.DataFrame* – pandas DataFrame object containing metadata on all edits in given modification
    """
    assert commit.merge
    # With merges, the following cases can occur:
    #   1. Changes of one or more parents are accepted.
    #   2. Changes made prior to the merge are replaced with new edits.
    # To obtain the state of the file before merging, get blame is executed on
    # all parent commits.
    try:
        file_content = git.Git(str(git_repo.path)) \
                          .show('{}:{}'.format(commit.hash,
                                               modification_info['new_path']))
    except GitCommandError:
        file_content = ''

    file_content_parents = []
    for parent in commit.parents:
        try:
            file_content_parents.append(
                git.Git(str(git_repo.path))
                   .show('{}:{}'.format(parent,
                                        modification_info['old_path'])))
        except GitCommandError:
            file_content_parents.append('')

    binary_file = is_binary_file(modification_info['new_path'], file_content)
    if not binary_file:
        for file_content_parent in file_content_parents:
            if is_binary_file(modification_info['new_path'],
                              file_content_parent):
                binary_file = True
                break

    if binary_file:
        blame_info_parent = None
        blame_info_commit = None
        added_lines = []
        deleted_lines = []

        edits = pd.DataFrame({'pre_start': None,
                              'number_of_deleted_lines': None,
                              'post_start': None,
                              'number_of_added_lines': None,
                              'type': 'binary_file_change'}, index=[0])

        edits_info = pd.DataFrame()
        for _, edit in edits.iterrows():
            e = {}
            e['commit_hash'] = commit.hash
            e['edit_type'] = edit.type
            e['total_added_lines'] = None
            e['total_removed_lines'] = None
            e.update(modification_info)

            e.update(_get_edit_details(edit, commit, deleted_lines,
                                       added_lines, blame_info_parent,
                                       blame_info_commit, extraction_settings))

            edits_info = pd.concat([edits_info, pd.DataFrame([e])],
                                   ignore_index=True, sort=False)

        return edits_info
    else:
        parent_blames = []
        for parent in commit.parents:
            try:
                parent_blame = git.Git(str(git_repo.path)) \
                                  .blame(parent,
                                         extraction_settings['blame_options'],
                                         modification_info['old_path'])

                if len(parent_blame) > 0:
                    parent_blame = _parse_porcelain_blame(parent_blame).rename(
                                   columns={'line_content': 'pre_line_content',
                                            'line_number': 'pre_line_number'})
                    parent_blame.loc[:, 'pre_commit'] = parent
                else:
                    parent_blame = pd.DataFrame({'original_commit_hash': [],
                                                 'original_line_no': [],
                                                 'original_file_path': [],
                                                 'pre_line_content': [],
                                                 'pre_line_number': [],
                                                 'pre_commit': []})
            except GitCommandError:
                parent_blame = pd.DataFrame({'original_commit_hash': [],
                                             'original_line_no': [],
                                             'original_file_path': [],
                                             'pre_line_content': [],
                                             'pre_line_number': [],
                                             'pre_commit': []})
            parent_blames.append(parent_blame)

    # Then, the current state of the file is obtained by executing git blame on
    # the current commit.
    try:
        current_blame = git.Git(str(git_repo.path)) \
                           .blame(commit.hash,
                                  extraction_settings['blame_options'],
                                  modification_info['new_path'])

        if len(current_blame) > 0:
            current_blame = _parse_porcelain_blame(current_blame).rename(
                            columns={'line_content': 'post_line_content',
                                     'line_number': 'post_line_number'})
        else:
            current_blame = pd.DataFrame({'original_commit_hash': [],
                                          'original_line_no': [],
                                          'original_file_path': [],
                                          'post_line_content': [],
                                          'post_line_number': []})
    except GitCommandError:
        current_blame = pd.DataFrame({'original_commit_hash': [],
                                      'original_line_no': [],
                                      'original_file_path': [],
                                      'post_line_content': [],
                                      'post_line_number': []})

    # Define columns that are considered when identifying duplicates.
    comp_cols = ['original_commit_hash', 'original_line_no',
                 'original_file_path']

    for idx, parent_blame in enumerate(parent_blames):
        parent_blames[idx]['_count'] = parent_blame.groupby(comp_cols) \
                                                   .cumcount()
    current_blame['_count'] = current_blame.groupby(comp_cols).cumcount()

    deletions = []
    additions = []
    for parent_blame in parent_blames:
        comp = parent_blame.merge(current_blame, on=comp_cols+['_count'],
                                  how='outer', indicator=True)
        comp['_action'] = None

        comp.loc[comp['_merge'] == 'both', '_action'] = 'accepted'
        comp.loc[comp['_merge'] == 'right_only', '_action'] = 'added'
        comp.loc[comp['_merge'] == 'left_only', '_action'] = 'deleted'

        assert not comp['_action'].isnull().any()

        drop_cols = ['_count', '_merge', '_action']

        added = comp.loc[comp['_action'] == 'added'].drop(drop_cols, axis=1)
        deleted = comp.loc[comp['_action'] == 'deleted'].drop(drop_cols,
                                                              axis=1)

        additions.append(added)
        deletions.append(deleted)

    added_lines_counter = collections.Counter()
    for added in additions:
        for _, x in added.iterrows():
            added_lines_counter[(x.post_line_number, x.post_line_content)] += 1

    added_lines = {k[0]: k[1] for k, v in added_lines_counter.items()
                   if v == len(commit.parents)}

    deleted_lines_parents = []
    for deleted in deletions:
        deleted_lines_parents.append({x.pre_line_number: x.pre_line_content
                                      for _, x in deleted.iterrows()})

    matches = []
    for k, v in added_lines.items():
        for idx, deleted_lines in enumerate(deleted_lines_parents):
            if k in deleted_lines and v == deleted_lines[k]:
                del deleted_lines_parents[idx][k]
                matches.append(k)
    for k in set(matches):
        del added_lines[k]

    edits_info = []

    edits_parents = []
    for deleted_lines in deleted_lines_parents:
        _, edits = _identify_edits(deleted_lines, added_lines,
                                   extraction_settings)
        edits_parents.append(edits)

    for idx, edits in enumerate(edits_parents):
        for _, edit in edits.iterrows():

            # extract edit details for all edits if merge deletions are
            # extracted or the edit type is not a deletion
            if extraction_settings['extract_merge_deletions'] or \
               (edit.type != 'deletion'):
                e = {}
                # Extract general information.
                e['commit_hash'] = commit.hash
                e['edit_type'] = edit.type
                e.update(modification_info)
                e.update(_get_edit_details(edit, commit,
                                           deleted_lines_parents[idx],
                                           added_lines, parent_blames[idx],
                                           current_blame, extraction_settings))

                edits_info.append(e)

    return pd.DataFrame(edits_info)


def _get_edited_file_paths_since_split(git_repo, commit):
    """
    For a merge commit returns list of all files edited since the last
    creation of a new branch relevant for the merge.

    :param pydriller.Git git_repo: pydriller Git object
    :param pydriller.Commit commit: pydriller Commit object

    :return:
        *List* – list of paths to the edited files
    """

    def expand_dag(dag, leafs):
        """
        Expands a dag by adding the parents of a given set of nodes to the dag.

        :param pathpy.DAG dag: pathpy DAG object
        :param set leafs: set of nodes that are expanded

        :return:
            *pathpy.DAG* – the expanded pathpy DAG object
        """
        for node in leafs:
            parents = git_repo.get_commit(node).parents
            for parent in parents:
                dag.add_edge(node, parent)
        return dag

    def common_node_on_paths(paths):
        """
        Computes the overlap between given sets of nodes. Returns the nodes
        present in all sets.

        :param List paths: list of node sequences

        :return:
            *set* – set of nodes that are present on all paths
        """
        # Drop first and last element of the path.
        common_nodes = set(paths[0][1:-1])
        for path in paths[1:]:
            common_nodes.intersection_update(path[1:-1])
        common_nodes = list(common_nodes)
        return common_nodes

    def remove_successors(dag, node):
        """
        Removes all successors of a node from a given dag.

        :param pathpy.DAG dag: pathpy DAG object
        :param str node: node for which successors shall be removed

        :return:
            *pathpy.DAG* – reduced pathpy DAG object
        """
        rm = [n for nl in [x[1:] for x in dag.routes_from_node(node)]
              for n in nl]
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

        paths = [p for pl in [dag.routes_to_node(node) for node in leafs]
                 for p in pl]
        common_nodes = common_node_on_paths(paths)

        if (len(leafs) == 1) or (len(common_nodes) > 0):
            cont = False

    for node in common_nodes:
        dag = remove_successors(dag, node)

    edited_file_paths = []
    for node in dag.nodes:
        edited_file_paths += [modification.new_path for modification
                              in git_repo.get_commit(node).modified_files]
        edited_file_paths += [modification.old_path for modification
                              in git_repo.get_commit(node).modified_files]

    edited_file_paths = set(edited_file_paths)
    if None in edited_file_paths:
        edited_file_paths.remove(None)

    return edited_file_paths


def _check_mailmap(name, email, git_repo):
    """
    Returns matching user from git .mailmap file if available. Returns
    input if user is not in mailmap or mailmap is unavailable.

    'param str name: username
    'param str email: git email
    'param pydriller.Git git_repo: PyDriller Git object

    :return:
        - *str* – corresponding username from mailmap
        - *str* – corresponding email from mailmap
    """
    if name.strip().startswith('-'):
        test_str = '<{}>'.format(email)
    else:
        test_str = '{} <{}>'.format(name, email)
    out_str = git.Git(str(git_repo.path)).check_mailmap(test_str)

    matches = re.findall("^(.*) <(.*)>$", out_str)
    if len(matches) > 1:
        raise Exception(("Error in mailmap check. Please report on "
                         "https://github.com/gotec/git2net."))
    elif len(matches) == 1:
        name, email = matches[0]
    # else name and email remain the same as the ones passed

    return name, email


def _format_edits_df(edits_df):
    """
    Formats the edits dataframe to the desired column order and data types.

    :param pandas.DataFrame edits_df: dataframe with edits

    :return:
        *pandas.DataFrame* – formatted dataframe
    """
    column_types = {
        'commit_hash': 'str',
        'edit_type': 'str',
        'filename': 'str',
        'levenshtein_dist': 'float',
        'modification_type': 'str',
        'new_path': 'str',
        'old_path': 'str',
        'original_commit_addition': 'str',
        'original_commit_deletion': 'str',
        'original_file_path_addition': 'str',
        'original_file_path_deletion': 'str',
        'original_line_no_addition': 'str',
        'original_line_no_deletion': 'str',
        'post_entropy': 'float',
        'post_len_in_chars': 'float',
        'post_len_in_lines': 'float',
        'post_starting_line_no': 'float',
        'pre_entropy': 'float',
        'pre_len_in_chars': 'float',
        'pre_len_in_lines': 'float',
        'pre_starting_line_no': 'float',
        'total_added_lines': 'float',
        'total_removed_lines': 'float'
    }
    # Ensure that all columns are present. If not, add them with None values.
    for column in column_types.keys():
        if column not in edits_df.columns:
            edits_df[column] = None

    edits_df = edits_df.astype(column_types)
    edits_df = edits_df[sorted(edits_df.columns)]

    return edits_df


def _process_commit(args):
    """
    Extracts information on commit and all edits made with the commit. As this function is run in
    parallel workers, any outputs are returned and output in a serial manner.

    :param dict args: dictionary with arguments. For multiprocessing, function can only
                      take single input.
                      Dictionary must contain:
                          git_repo_dir: path to the git repository that is mined
                          commit_hash: hash of the commit that is processed
                          extraction_settings: settings for the extraction

    :return:
        *dict* – dict containing two dataframes with information of commit and edits
        *tuple* – tuple of logging type and logging message if any were created, otherwise None
        *str* – text of exception if any was raised, otherwise None
    """
    try:
        log = None
        exception = None

        with contextlib.redirect_stderr(io.StringIO()) as redirected_stderr_ctx_mgr:
            with git_init_lock:
                git_repo = pydriller.Git(args['git_repo_dir'])
                commit = git_repo.get_commit(args['commit_hash'])

            with Timeout(args['extraction_settings']['timeout']) as timeout:
                author_name, author_email = _check_mailmap(commit.author.name,
                                                           commit.author.email,
                                                           git_repo)

                committer_name, committer_email = _check_mailmap(commit.committer.name,
                                                                 commit.committer.email,
                                                                 git_repo)

                # parse commit
                c = {}
                c['hash'] = commit.hash
                c['author_email'] = author_email
                c['author_name'] = author_name
                c['committer_email'] = committer_email
                c['committer_name'] = committer_name
                c['author_date'] = commit.author_date.strftime('%Y-%m-%d %H:%M:%S')
                c['committer_date'] = commit.committer_date \
                                            .strftime('%Y-%m-%d %H:%M:%S')
                c['author_timezone'] = commit.author_timezone
                c['committer_timezone'] = commit.committer_timezone
                c['no_of_modifications'] = len(commit.modified_files)
                c['commit_message_len'] = len(commit.msg)
                if args['extraction_settings']['extract_text']:
                    c['commit_message'] = commit.msg \
                                                .encode('utf8', 'surrogateescape') \
                                                .decode('utf8', 'replace')
                c['project_name'] = commit.project_name
                c['parents'] = ','.join(commit.parents)
                c['merge'] = commit.merge
                c['in_main_branch'] = commit.in_main_branch
                c['branches'] = ','.join(commit.branches)

                # parse modification
                df_edits_list = []
                if commit.merge and args['extraction_settings']['extract_merges']:
                    # Git does not create a modification if own changes are accpeted
                    # during a merge. Therefore, the edited files are extracted
                    # manually.
                    edited_file_paths = {f for p in commit.parents for f in
                                         git.Git(str(git_repo.path))
                                            .diff(commit.hash, p, '--name-only')
                                            .split('\n')}

                    if (args['extraction_settings']['max_modifications'] > 0) and \
                       (len(edited_file_paths) >
                            args['extraction_settings']['max_modifications']):
                        log = ('warning', 'max_modifications exceeded: ' + commit.hash)
                        extracted_result = {'commit': pd.DataFrame(),
                                            'edits': pd.DataFrame()}
                        return extracted_result, log, exception

                    for edited_file_path in edited_file_paths:
                        exclude_file = False
                        for x in args['extraction_settings']['exclude']:
                            if edited_file_path.startswith(x + os.sep) or \
                               (edited_file_path == x):
                                exclude_file = True
                        if not exclude_file:
                            modification_info = {}
                            try:
                                file_content = git.Git(  # noqa
                                    str(git_repo.path)
                                ).show('{}:{}'.format(commit.hash, edited_file_path))

                                modification_info['filename'] = edited_file_path.split(os.sep)[-1]
                                modification_info['new_path'] = edited_file_path
                                modification_info['old_path'] = edited_file_path
                                modification_info['modification_type'] = 'merge_self_accept'

                                df_edits_list.append(
                                     _extract_edits_merge(
                                         git_repo, commit, modification_info,
                                         args['extraction_settings'])
                                )
                            except GitCommandError:
                                # A GitCommandError occurs if the file was deleted. In
                                # this case it currently has no content.

                                # Get filenames from all modifications in merge commit.
                                paths = [m.old_path for m in commit.modified_files]

                                # Analyse changes if modification was recorded. Else,
                                # the deletions were made before the merge.
                                if edited_file_path in paths:
                                    modification_info['filename'] = edited_file_path.split(os.sep)[-1]
                                    # File was deleted.
                                    modification_info['new_path'] = None
                                    modification_info['old_path'] = edited_file_path
                                    modification_info['modification_type'] = 'merge_self_accept'

                                    df_edits_list.append(
                                         _extract_edits_merge(
                                             git_repo, commit, modification_info,
                                             args['extraction_settings'])
                                    )
                else:
                    if (args['extraction_settings']['max_modifications'] > 0) and \
                       (len(commit.modified_files) >
                           args['extraction_settings']['max_modifications']):
                        log = ('warning', 'max_modifications exceeded: ' + commit.hash)
                        extracted_result = {'commit': pd.DataFrame(),
                                            'edits': pd.DataFrame()}
                        return extracted_result, log, exception

                    for modification in commit.modified_files:
                        exclude_file = False
                        for x in args['extraction_settings']['exclude']:
                            if modification.new_path:
                                if modification.new_path.startswith(x + os.sep) or \
                                   (modification.new_path == x):
                                    exclude_file = True
                            if not exclude_file and modification.old_path:
                                if modification.old_path.startswith(x + os.sep):
                                    exclude_file = True
                        if not exclude_file:
                            df_edits_list.append(
                                _extract_edits(
                                    git_repo, commit, modification,
                                    args['extraction_settings'])
                            )

                # Remove empty dataframes, sort the columns, and align the data types to ensure
                # compatibility with the latest pandas version.
                df_edits_list = [_format_edits_df(x) for x in df_edits_list if not x.empty]
                if len(df_edits_list) == 0:
                    df_edits = pd.DataFrame()
                else:
                    df_edits = pd.concat(
                        df_edits_list,
                        axis=0, ignore_index=True, sort=True
                    )

                df_commit = pd.DataFrame(c, index=[0])

                extracted_result = {'commit': df_commit, 'edits': df_edits}

        redirected_stderr = redirected_stderr_ctx_mgr.getvalue().strip()

        if redirected_stderr:
            # Something was written to stderr. This could be a real error, however, often
            # it is just a notification that an exception was ignored while deleting an object
            # after the threading timeout triggered. In this case, we ignore the message.
            if not (redirected_stderr.startswith('Exception ignored in:') and
                    redirected_stderr.endswith('git2net.extraction.TimeoutException:')):
                extracted_result = {'commit': pd.DataFrame(), 'edits': pd.DataFrame()}
                log = ('error', 'processing error: ' + commit.hash)
                exception = exception

        if pd.isnull(exception) and timeout.timed_out:
            log = ('warning', 'processing timeout: ' + commit.hash)
            extracted_result = {'commit': pd.DataFrame(), 'edits': pd.DataFrame()}

        return extracted_result, log, exception

    except Exception as e:
        return e


def _log_commit_results(log, exception):
    """
    Processes log and exception returned from _process_commit and creates corresponding entries.

    :param tuple log: tuple of logging type and logging message if any were created, otherwise None
    :param str exception: text of exception if any was raised, otherwise None

    :return:
        log message will be written and exception raised if one occurred
    """

    LOG = logging.getLogger('git2net')

    if pd.notnull(log):
        if log[0] == 'warning':
            LOG.warning(log[1])
        elif log[0] == 'error':
            LOG.error(log[1])
            raise Exception(exception)
        else:
            Exception(("Not implemented logging type. Please report on "
                       "https://github.com/gotec/git2net."))


def _process_repo_serial(git_repo_dir, sqlite_db_file, commits,
                         extraction_settings):
    """
    Processes all commits in a given git repository in a serial manner.

    :param str git_repo_dir: path to the git repository that is mined
    :param str sqlite_db_file: path (including database name) where the sqlite database will be created
    :param List[str] commits: list of commits that have to be processed
    :param dict extraction_settings: settings for the extraction

    :return:
        SQLite database will be written at specified location
    """

    LOG = logging.getLogger('git2net')  # noqa

    for commit in tqdm(commits, desc='Serial'):
        with logging_redirect_tqdm(tqdm_class=tqdm):
            args = {'git_repo_dir': git_repo_dir, 'commit_hash': commit.hash,
                    'extraction_settings': extraction_settings}
            res = _process_commit(args)

            if isinstance(res, Exception):
                raise res
            else:
                extracted_result, log, exception = res
                _log_commit_results(log, exception)

                with sqlite3.connect(sqlite_db_file) as con:
                    if not extracted_result['edits'].empty:
                        extracted_result['edits'].to_sql('edits', con, if_exists='append',
                                                         index=False)
                    if not extracted_result['commit'].empty:
                        extracted_result['commit'].to_sql('commits', con, if_exists='append',
                                                          index=False)


# suggestion by marco-c (github.com/ishepard/pydriller/issues/110)
def _init(git_repo_dir, git_init_lock_):
    global git_init_lock
    git_init_lock = git_init_lock_


def _process_repo_parallel(git_repo_dir, sqlite_db_file, commits,
                           extraction_settings):
    """
    Processes all commits in a given git repository in a parallel manner.

    :param str git_repo_dir: path to the git repository that is mined
    :param str sqlite_db_file: path (including database name) where the sqlite database will be created
    :param List[str] commits: list of commits that are already in the database
    :param dict extraction_settings: settings for the extraction

    :return:
        SQLite database will be written at specified location
    """

    LOG = logging.getLogger('git2net')  # noqa

    args = [{'git_repo_dir': git_repo_dir, 'commit_hash': commit.hash,
             'extraction_settings': extraction_settings}
            for commit in commits]

    with multiprocessing.Pool(extraction_settings['no_of_processes'],
                              initializer=_init,
                              initargs=(git_repo_dir, git_init_lock)) as p:
        with tqdm(total=len(args), desc='Parallel ({0} processes)'
                  .format(extraction_settings['no_of_processes'])) as pbar:
            with logging_redirect_tqdm(tqdm_class=tqdm):
                for res in p.imap_unordered(_process_commit, args, chunksize=extraction_settings['chunksize']):

                    if isinstance(res, Exception):
                        raise res
                    else:
                        extracted_result, log, exception = res

                        _log_commit_results(log, exception)

                        with sqlite3.connect(sqlite_db_file) as con:
                            if not extracted_result['edits'].empty:
                                extracted_result['edits'].to_sql('edits', con,
                                                                 if_exists='append', index=False)
                            if not extracted_result['commit'].empty:
                                extracted_result['commit'].to_sql('commits', con,
                                                                  if_exists='append',
                                                                  index=False)
                        pbar.update(1)


def identify_file_renaming(git_repo_dir):
    """
    Identifies all names and locations different files in a repository have had.

    :param str git_repo_dir: path to the git repository that is mined

    :return:
        - *pathpy.DAG* – pathpy DAG object depicting the renaming process
        - *dict* – dictionary containing all aliases for all files
    """

    # TODO: Consider corner case where file is renamed and new file with old
    # name is created.
    git_repo = pydriller.Git(git_repo_dir)

    dag = pp.DAG()
    for commit in tqdm(list(git_repo.get_list_commits()), desc='Creating DAG'):
        for modification in commit.modified_files:

            if (modification.new_path not in dag.nodes) and \
               (modification.old_path == modification.new_path) and \
               (modification.change_type == pydriller.domain.commit
                                                     .ModificationType.ADD):
                if modification.new_path not in dag.nodes:
                    dag.add_node(modification.new_path)
            elif modification.old_path != modification.new_path:
                if pd.isnull(modification.old_path):
                    if modification.new_path not in dag.nodes:
                        dag.add_node(modification.new_path)
                elif pd.isnull(modification.new_path):
                    pass
                else:
                    dag.add_edge(modification.new_path, modification.old_path)

    dag.make_acyclic()
    nodes = [k for k, v in dag.nodes.items() if v['indegree'] == 0 and
             not v['outdegree'] == 0]
    aliases = {z: y[-1] for x in nodes for y in dag.routes_from_node(x) for z
               in y[:-1]}

    return dag, aliases


def get_unified_changes(git_repo_dir, commit_hash, file_path):
    """
    Returns dataframe with github-like unified diff representation of the
    content of a file before and after a commit for a given git repository,
    commit hash and file path.

    :param str git_repo_dir: path to the git repository that is mined
    :param str commit_hash: commit hash for which the changes are computed
    :param str file_path: path to file (within the repository) for which the changes are computed

    :return:
        *pandas.DataFrame* – pandas dataframe listing changes made to file in commit
    """
    git_repo = pydriller.Git(git_repo_dir)
    commit = git_repo.get_commit(commit_hash)

    # Select the correct modifictaion.
    for modification in commit.modified_files:
        if modification.new_path == file_path:
            break

    # Parse the diff extracting the lines added and deleted with the given
    # commit.
    parsed_lines = modification.diff_parsed

    deleted_lines = {x[0]: x[1] for x in parsed_lines['deleted']}
    added_lines = {x[0]: x[1] for x in parsed_lines['added']}

    # Indetify the edits made with the changes.
    pre_to_post, edits = _identify_edits(deleted_lines, added_lines,
                                         {'use_blocks': False})

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
    while (post_counter < len(post_source_code)) or \
          (pre_counter < max(deleted_lines.keys())) or \
          (post_counter < max(added_lines.keys())):
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

    df = pd.DataFrame({'pre': pre_line_no, 'post': post_line_no,
                       'action': action, 'code': code})

    return df


def check_mining_complete(git_repo_dir, sqlite_db_file, commits=[],
                          all_branches=False, return_number_missing=False):
    """
    Checks status of a mining operation

    :param str git_repo_dir: path to the git repository that is mined
    :param str sqlite_db_file: path (including database name) where with sqlite database
    :param List[str] commits: only consider specific set of commits, considers all if empty

    :return:
        *bool* – True if all commits are included in the database, otherwise False
    """
    LOG = logging.getLogger('git2net')

    git_repo = pydriller.Git(git_repo_dir)
    if os.path.exists(sqlite_db_file):
        try:
            with sqlite3.connect(sqlite_db_file) as con:
                try:
                    p_commits = set(x[0] for x in
                                    con.execute("SELECT hash FROM commits")
                                    .fetchall())
                except sqlite3.OperationalError:
                    LOG.warning("Found a database on provided path that did not contain a 'commits' table.")
                    p_commits = set()
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            raise Exception("The provided file is not a compatible database.")
    else:
        raise Exception("Found no database at provided path.")

    if not commits:
        commits = [c.hash for c in git_repo.get_list_commits(all=all_branches)]
    if set(commits).issubset(p_commits):
        if return_number_missing:
            return (True, 0)
        else:
            return True
    else:
        if return_number_missing:
            return (False, len(set(commits).difference(p_commits)))
        else:
            return False


def mining_state_summary(git_repo_dir, sqlite_db_file, all_branches=False):
    """
    Prints mining progress of database and returns dataframe with details
    on missing commits.

    :param str git_repo_dir: path to the git repository that is mined
    :param str sqlite_db_file: path (including database name) where with sqlite database

    :return:
        *pandas.DataFrame* – dataframe with details on missing commits
    """
    LOG = logging.getLogger('git2net')

    git_repo = pydriller.Git(git_repo_dir)
    if os.path.exists(sqlite_db_file):
        try:
            with sqlite3.connect(sqlite_db_file) as con:
                try:
                    p_commits = set(x[0] for x in
                                    con.execute("SELECT hash FROM commits")
                                    .fetchall())
                except sqlite3.OperationalError:
                    LOG.warning("Found a database on provided path that did not contain a 'commits' table.")
                    p_commits = set()
        except sqlite3.OperationalError:
            raise Exception("The provided file is not a compatible database.")
    else:
        raise Exception("Found no database at provided path.")

    commits = [c for c in git_repo.get_list_commits(all=all_branches)]
    if not p_commits.issubset({c.hash for c in commits}):
        raise Exception("The database does not match the provided repository.")

    no_of_commits = len({c.hash for c in commits})
    LOG = logging.getLogger('git2net')
    LOG.info('{} / {} ({:.2f}%) of commits were successfully mined.'.format(
        len(p_commits), no_of_commits,
        len(p_commits) / no_of_commits * 100))

    u_commits = [c for c in commits if c.hash not in p_commits]

    u_commit_info = {'hash': [],
                     'is_merge': [],
                     'modifications': [],
                     'author_name': [],
                     'author_email': [],
                     'author_date': []}
    for c in tqdm(u_commits):
        u_commit_info['hash'].append(c.hash)
        try:
            u_commit_info['is_merge'].append(c.merge)
        except Exception as e:
            LOG.error(f'Error {e} reading "merge" for', c.hash)
            u_commit_info['is_merge'].append(None)

        if c.merge:
            u_commit_info['modifications'].append(
                len({f for p in c.parents for f in
                     git.Git(str(git_repo.path))
                        .diff(c.hash, p, '--name-only').split('\n')}))
        else:
            u_commit_info['modifications'].append(len(c.modified_files))

        try:
            u_commit_info['author_name'].append(c.author.name)
        except Exception as e:
            LOG.error(f'Error {e} reading "author.name" for', c.hash)
            u_commit_info['author_name'].append(None)

        try:
            u_commit_info['author_email'].append(c.author.email)
        except Exception as e:
            LOG.error(f'Error {e} reading "author.email" for', c.hash)
            u_commit_info['author_email'].append(None)

        try:
            u_commit_info['author_date'].append(
                c.author_date.strftime('%Y-%m-%d %H:%M:%S'))
        except Exception as e:
            LOG.error(f'Error {e} reading "author_date" for', c.hash)
            u_commit_info['author_date'].append(None)

    u_commits_info = pd.DataFrame(u_commit_info)

    return u_commits_info


def mine_git_repo(git_repo_dir, sqlite_db_file, commits=[],
                  use_blocks=False, no_of_processes=os.cpu_count(),
                  chunksize=1, exclude=[], blame_C='', blame_w=False,
                  max_modifications=0, timeout=0, extract_text=False,
                  extract_merges=True, extract_merge_deletions=False,
                  all_branches=False):
    """
    Creates sqlite database with details on commits and edits for a given
    git repository.

    :param str git_repo_dir: path to the git repository that is mined
    :param str sqlite_db_file: path (including database name) where the sqlite database will be created
    :param List[str] commits: only consider specific set of commits, considers all if empty
    :param bool use_blocks: determins if analysis is performed on block or line basis
    :param int no_of_processes: number of parallel processes that are spawned
    :param int chunksize: number of tasks that are assigned to a process at a time
    :param List[str] exclude: file paths that are excluded from the analysis
    :param str blame_C: string for the blame C option following the pattern "-C[<num>]" (computationally expensive)
    :param bool blame_w: bool, ignore whitespaces in git blame (-w option)
    :param int max_modifications: ignore commit if there are more modifications
    :param int timeout: stop processing commit after given time in seconds
    :param bool extract_text: extract the commit message and line texts
    :param bool extract_merges: process merges
    :param bool extract_merge_deletions: extract lines that are not accepted during a merge as 'deletions'

    :return:
        SQLite database will be written at specified location
    """
    LOG = logging.getLogger('git2net')

    git_version = check_output(['git', '--version']).strip().decode("utf-8")

    parsed_git_version = re.search(r'(\d+)\.(\d+)\.(\d+)', git_version) \
                           .groups()

    if int(parsed_git_version[0]) < 2 or \
       (
            (int(parsed_git_version[0]) == 2) and
            (
                (int(parsed_git_version[0]) == 2) and
                (int(parsed_git_version[1]) < 11)
            ) or
            (int(parsed_git_version[1]) == 11) and
            (int(parsed_git_version[2]) == 0)
       ):
        raise Exception("Your system is using " + git_version +
                        " which is not supported by git2net. " +
                        " Please update to git >= 2.11.1")

    blame_options = _parse_blame_C(blame_C) + ['--show-number',
                                               '--line-porcelain']
    if blame_w:
        blame_options += ['-w']

    extraction_settings = {'use_blocks': use_blocks,
                           'no_of_processes': no_of_processes,
                           'chunksize': chunksize,
                           'exclude': exclude,
                           'blame_options': blame_options,
                           'max_modifications': max_modifications,
                           'timeout': timeout,
                           'extract_text': extract_text,
                           'extract_merges': extract_merges,
                           'extract_merge_deletions': extract_merge_deletions}

    git_repo = pydriller.Git(git_repo_dir)

    if os.path.exists(sqlite_db_file):
        try:
            with sqlite3.connect(sqlite_db_file) as con:
                prev_method, prev_repository, prev_extract_text = \
                    con.execute("""SELECT method,
                                          repository,
                                          extract_text
                                   FROM _metadata""").fetchall()[0]

                if (prev_method == 'blocks' if use_blocks else 'lines') and \
                   (prev_extract_text == str(extract_text)):
                    try:
                        # processed commits
                        p_commits = set(x[0] for x in
                                        con.execute("SELECT hash FROM commits")
                                        .fetchall())
                    except sqlite3.OperationalError:
                        p_commits = set()
                    if all_branches:
                        c_commits = set(c.hash for c in
                                        git_repo.get_list_commits(all=True))
                    else:
                        c_commits = set(c.hash for c in
                                        git_repo.get_list_commits())
                    if not p_commits.issubset(c_commits):
                        if prev_repository != git_repo.repo.remotes.origin.url:
                            raise Exception(("Found a database that was "
                                             "created with identical settings. "
                                             "However, some commits in the "
                                             "database are not in the provided "
                                             "git repository and the url of "
                                             "the current origin is different "
                                             "to the one listed in the "
                                             "database. Please provide a clean "
                                             "database or update the origin in "
                                             "the current database if you want "
                                             "to proceed."))
                    else:
                        if p_commits == c_commits:
                            LOG.info("All commits have already been mined!")
                        else:
                            LOG.info('Found a matching database on provided path.')
                            LOG.info('\t Skipping {} ({:.2f}%) of {} commits.'
                                     .format(len(p_commits),
                                             len(p_commits)/len(c_commits)*100,
                                             len(c_commits)))
                            LOG.info('\t {} commits remaining.'
                                     .format(len(c_commits)-len(p_commits)))
                else:
                    raise Exception(("Found a database on provided path that "
                                     "was created with settings not matching "
                                     "the ones selected for the current run. A "
                                     "path to either no database or a database "
                                     "from a previously paused run with "
                                     "identical settings is required."))
        except sqlite3.OperationalError:
            raise Exception(("Found a database on provided path that was "
                             "likely not created with git2net. A path to "
                             "either no database or a database from a "
                             "previously paused run with identical settings "
                             "is required."))
    else:
        LOG.info("Found no database on provided path. Starting from scratch.")
        try:
            repo_url = git_repo.repo.remotes.origin.url
        except Exception as e:
            LOG.warning(f"Error {e} reading repository url.")
            repo_url = git_repo_dir

        with sqlite3.connect(sqlite_db_file) as con:
            con.execute("""CREATE TABLE _metadata ('created with',
                                                   'repository',
                                                   'date',
                                                   'method',
                                                   'extract_text')""")
            con.execute("""INSERT INTO _metadata ('created with',
                                                  'repository',
                                                  'date',
                                                  'method',
                                                  'extract_text')
                        VALUES (:version,
                                :repository,
                                :date,
                                :method,
                                :extract_text)""",
                        {'version': 'git2net ' + str(__version__),
                         'repository': repo_url,
                         'date': datetime.datetime.now()
                                         .strftime('%Y-%m-%d %H:%M:%S'),
                         'method': 'blocks' if use_blocks else 'lines',
                         'extract_text': str(extract_text)})
            con.commit()
            p_commits = set()

    # commits in the currently mined repository
    if all_branches:
        c_commits = set(c.hash for c in
                        git_repo.get_list_commits(all=True))
    else:
        c_commits = set(c.hash for c in
                        git_repo.get_list_commits())

    if not commits:
        # unprocessed commits
        if all_branches:
            u_commits = [c for c in git_repo.get_list_commits(all=True) if
                         c.hash not in p_commits]
        else:
            u_commits = [c for c in git_repo.get_list_commits() if c.hash not
                         in p_commits]
    else:
        if not set(commits).issubset(c_commits):
            raise Exception(("At least one provided commit does not exist in the repository."))
        commits = [git_repo.get_commit(h) for h in commits]
        u_commits = [c for c in commits if c.hash not in p_commits]

    # Add information on commit being present in currently active branch
    current_branch = git_repo.repo.active_branch.name
    c_intersect = p_commits.intersection(c_commits)
    if len(c_intersect) > 0:
        LOG.info(("Updated branch information for mined commits in the active branch."))
        for c in p_commits.intersection(c_commits):
            b = con.execute("""SELECT branches
                               FROM commits
                               WHERE hash = (:hash)""",
                            {'hash': c}).fetchall()[0][0]
            b = set(b.split(','))
            if current_branch not in b:
                b.add(current_branch)
                con.execute("""UPDATE commits
                               SET branches = (:branches)
                               WHERE hash = (:hash)""",
                            {'branches': ','.join(b), 'hash': c})
        con.commit()

    if len(u_commits) > 0:
        if extraction_settings['no_of_processes'] > 1:
            _process_repo_parallel(git_repo_dir, sqlite_db_file, u_commits,
                                   extraction_settings)
        else:
            _process_repo_serial(git_repo_dir, sqlite_db_file, u_commits,
                                 extraction_settings)


def mine_github(github_url, git_repo_dir, sqlite_db_file, branch=None,
                **kwargs):
    """
    Clones a repository from github and starts the mining process.

    :param str github_url: url to the publicly accessible github project that will be mined can be
        priovided as full url or <OWNER>/<REPOSITORY>
    :param str git_repo_dir: path to the git repository that is mined if path ends with '/' an
        additional folder will be created
    :param str sqlite_db_file: path (including database name) where the sqlite database will be created
    :param str branch: The branch of the github project that will be checked out and mined. If no
        branch is provided the default branch of the repository is used.
    :param **kwargs: arguments that will be passed on to mine_git_repo

    :return:
        - git repository will be cloned to specified location
        - SQLite database will be written at specified location
    """  # noqa
    LOG = logging.getLogger('git2net')

    # github_url can either be provided as full url or as in form <USER>/<REPO>
    user_repo_pattern = r'^([^\/]*)\/([^\/]*)$'
    full_url_pattern = r'^https:\/\/github\.com\/([^\/]*)\/([^\/.]*)(\.git)?$'

    match = re.match(user_repo_pattern, github_url)
    if match:
        git_owner = match.groups()[0]
        git_repo = match.groups()[1]
        github_url = 'https://github.com/{}/{}'.format(git_owner, git_repo)
    else:
        match = re.match(full_url_pattern, github_url)
        if match:
            git_owner = match.groups()[0]
            git_repo = match.groups()[1]
        else:
            raise Exception('Invalid github_url provided.')

    # detect the correct directories to work with
    local_directory = '/'.join(git_repo_dir.split('/')[:-1])
    git_repo_folder = git_repo_dir.split('/')[-1]

    if local_directory == '':
        local_directory = '.'
    if git_repo_folder == '':
        git_repo_folder = git_owner + '__' + git_repo

    # check if the folder is empty if it exists
    if os.path.exists(git_repo_dir) and \
       (len(os.listdir(os.path.join(local_directory, git_repo_folder))) > 0):
        LOG.info('Provided folder is not empty.')
        LOG.info('\t Skipping the cloning and trying to resume.')
    else:
        if branch:
            git.Git(local_directory).clone(github_url, git_repo_folder,
                                           branch=branch)
        else:
            git.Git(local_directory).clone(github_url, git_repo_folder)

    # mine the cloned repo
    mine_git_repo(git_repo_dir, sqlite_db_file, **kwargs)
