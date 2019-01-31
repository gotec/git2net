#!/usr/bin/python3

import logging
import sqlite3
import os
import argparse
import sys

from multiprocessing import Pool
from multiprocessing import Semaphore

import pandas as pd
import progressbar
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy

import pydriller as pydriller
from pydriller.git_repository import GitCommandError
from Levenshtein import distance as lev_dist

import pathpy as pp

logger = logging.getLogger(__name__)

def get_block_size(lines, k):
    """
    Calculates the length (in number of lines) of a block of added/deleted lines starting in a given
    line k.

    Parameters
    ----------
    @param lines: dictionary of added or deleted lines
    @param k: line number to check for
    """
    if k not in lines or (k > 1 and k - 1 in lines):
        block = False
        block_size = 0
    else:
        block = True
        block_size = 1

    while block:
        if k + block_size in lines:
            block_size += 1
        else:
            block = False
    return block_size

def identify_edits(deleted_lines, added_lines):
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
    # create block dictionary
    # keys are deleted line numbers
    # values are tuples containing
    #  * number of deleted lines
    #  * right line number
    #  * number of added lines
    blocks = {}
    right = min(min_added, min_deleted)
    left = min(min_added, min_deleted)

    no_right_inc = 0
    both_inc = 0
    jump_right = 0

    while left <= max(max_added, max_deleted):

        # compute size of added and deleted blocks
        added_block = get_block_size(added_lines, right)
        deleted_block = get_block_size(deleted_lines, left)

        # we ignore pure additions
        if deleted_block > 0:
            blocks[left] = (deleted_block, right, added_block)

        # deleted block is larger than added block
        if deleted_block > added_block:
            no_right_inc = deleted_block - added_block
            both_inc = added_block

        # added block is larger than deleted block
        elif added_block > deleted_block:
            jump_right = added_block - deleted_block
            both_inc = deleted_block

        # increment left and right counter
        if both_inc>0:
            both_inc -= 1
            pre_to_post[left] = right
            left +=1
            right += 1
        elif no_right_inc>0:
                no_right_inc -= 1
                pre_to_post[left] = False
                left += 1
        else:
            right += jump_right
            pre_to_post[left] = right
            left +=1
            right += 1
            jump_right = 0

    return pre_to_post, blocks

def text_entropy(text):
    return entropy([text.count(chr(i)) for i in range(256)], base=2)

def extract_coedits(git_repo, commit, mod, use_blocks=False):

    df = pd.DataFrame()

    path = mod.new_path

    parsed_lines = git_repo.parse_diff(mod.diff)

    deleted_lines = { x[0]:x[1] for x in parsed_lines['deleted'] }
    added_lines = { x[0]:x[1] for x in parsed_lines['added'] }

    pre_to_post, blocks = identify_edits(deleted_lines, added_lines)

    try:
        blame = git_repo.git.blame(commit.hash + '^', '--', path).split('\n')
        if use_blocks:
            for num_line, line in deleted_lines.items():
                if num_line in blocks.keys():
                    block = blocks[num_line]

                    blame_fields = blame[num_line - 1].split(' ')
                    buggy_commit_hash = blame_fields[0].replace('^', '')

                    c = {}
                    c['mod_filename'] = mod.filename
                    c['mod_new_path'] = path
                    c['mod_old_path'] = mod.old_path
                    c['pre_commit'] = buggy_commit_hash
                    c['post_commit'] = commit.hash
                    c['mod_added'] = mod.added
                    c['mod_removed'] = mod.removed
                    c['mod_cyclomatic_complexity'] = mod.complexity
                    c['mod_loc'] = mod.nloc
                    c['mod_token_count'] = mod.token_count

                    deleted_block = []
                    for i in range(num_line, num_line + block[0]):
                        deleted_block.append(deleted_lines[i])

                    added_block = []
                    for i in range(block[1], block[1] + block[2]):
                        added_block.append(added_lines[i])

                    deleted_block = ' '.join(deleted_block)
                    added_block = ' '.join(added_block)

                    c['pre_block_starting_line_num'] = num_line
                    c['pre_block_len_in_lines'] = block[0]
                    c['pre_block_len_in_chars'] = len(deleted_block)

                    c['post_block_starting_line_num'] = block[1]
                    c['post_block_len_in_lines'] = block[2]
                    c['post_block_len_in_chars'] = len(added_block)

                    if len(deleted_block) > 0:
                        c['pre_block_entropy'] = text_entropy(deleted_block)
                    else:
                        c['pre_block_entropy'] = None

                    if len(added_block) > 0:
                        c['post_block_entropy'] = text_entropy(added_block)
                    else:
                        c['post_block_entropy'] = None

                    # if no lines were added (only deletion)
                    if block[2] == 0:
                        c['levenshtein_dist'] = None
                    else:
                        c['levenshtein_dist'] = lev_dist(deleted_block, added_block)


                    df = df.append(c, ignore_index=True)
        else:
            for num_line, line in deleted_lines.items():
                blame_fields = blame[num_line - 1].split(' ')
                buggy_commit_hash = blame_fields[0].replace('^', '')

                c = {}
                c['mod_filename'] = mod.filename
                c['mod_new_path'] = path
                c['mod_old_path'] = mod.old_path
                c['pre_commit'] = buggy_commit_hash
                c['post_commit'] = commit.hash
                c['pre_line_num'] = num_line
                c['pre_line_len'] = len(line)
                c['mod_added'] = mod.added
                c['mod_removed'] = mod.removed
                c['mod_cyclomatic_complexity'] = mod.complexity
                c['mod_loc'] = mod.nloc
                c['mod_token_count'] = mod.token_count

                if len(line) > 0:
                    c['pre_line_entropy'] = text_entropy(line)
                else:
                    c['pre_line_entropy'] = None

                if pre_to_post[num_line] in added_lines:
                    c['post_line_num'] = pre_to_post[num_line]
                    c['post_line_len'] = len(added_lines[pre_to_post[num_line]])

                    if len(added_lines[pre_to_post[num_line]]) > 0:
                        c['post_line_entropy'] = text_entropy(added_lines[pre_to_post[num_line]])
                    else:
                        c['post_line_entropy'] = None

                    c['levenshtein_dist'] = lev_dist(added_lines[pre_to_post[num_line]], line)
                else:
                    c['post_line_len'] = 0
                    c['post_line_num'] = None
                    c['levenshtein_dist'] = None
                df = df.append(c, ignore_index=True)

    except GitCommandError:
        logger.debug("Could not found file %s in commit %s. Probably a double rename!",
                        mod.filename, commit.hash)

    return df


def process_commit(git_repo, commit, exclude_paths = set(), use_blocks = False):
    df_commit = pd.DataFrame()
    df_coedits = pd.DataFrame()

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
    c['modifications'] = len(commit.modifications)
    c['msg_len'] = len(commit.msg)
    c['project_name'] = commit.project_name
    c['parents'] = ','.join(commit.parents)
    c['merge'] = commit.merge
    c['in_main_branch'] = commit.in_main_branch
    c['branches'] = ','.join(commit.branches)

    df_commit = pd.DataFrame(c, index=[0])

    # parse modification
    for modification in commit.modifications:
        exclude_file = False
        excluded_path = ''
        for x in exclude_paths:
            if modification.new_path:
                if modification.new_path.startswith(x+os.sep):
                    exclude_file = True
                    excluded_path = modification.new_path
            if not exclude_file and modification.old_path:
                if modification.old_path.startswith(x+os.sep):
                    exclude_file = True
                    excluded_path = modification.old_path
        if not exclude_file:
            df = extract_coedits(git_repo, commit, modification, use_blocks=use_blocks)
            df_coedits = pd.concat([df_coedits, df])
    return df_commit, df_coedits



def process_repo_serial(repo_string, exclude=None, use_blocks=False):

    git_repo = pydriller.GitRepository(repo_string)
    exclude_paths = []
    if exclude:
        with open(exclude) as f:
            exclude_paths = [x.strip() for x in f.readlines()]

    df_commits = pd.DataFrame()
    df_coedits = pd.DataFrame()

    i = 0
    commits = [c for c in git_repo.get_list_commits()]
    for commit in tqdm(commits, desc='Serial'):
        df_1, df_2 = process_commit(git_repo, commit, exclude_paths, use_blocks=use_blocks)
        df_commits = pd.concat([df_commits, df_1])
        df_coedits = pd.concat([df_coedits, df_2])
    return df_commits, df_coedits

semaphore = Semaphore(1)


def process_commit_parallel(arg):
    repo_string, commit_hash, sqlite_db_file, exclude, use_blocks = arg
    git_repo = pydriller.GitRepository(repo_string)

    exclude_paths = []
    if exclude:
        with open(exclude) as f:
            exclude_paths = [x.strip() for x in f.readlines()]

    df_1, df_2 = process_commit(git_repo, git_repo.get_commit(commit_hash), exclude_paths,
                                use_blocks=use_blocks)

    with semaphore:
        con = sqlite3.connect(sqlite_db_file)
        if not df_1.empty:
            df_1.to_sql('commits', con, if_exists='append')
        if not df_2.empty:
            df_2.to_sql('coedits', con, if_exists='append')
    return True


def process_repo_parallel(repo_string, sqlite_db_file, num_processes=os.cpu_count(), exclude=None,
                          chunksize=1, use_blocks=False):

    git_repo = pydriller.GitRepository(repo_string)
    args = [ (repo_string, c.hash, sqlite_db_file, exclude, use_blocks)
            for c in git_repo.get_list_commits()]

    p = Pool(num_processes)
    with tqdm(total=len(args), desc='Parallel ({0} workers)'.format(num_processes)) as pbar:
        for i,_ in enumerate(p.imap_unordered(process_commit_parallel, args, chunksize=chunksize)):
            pbar.update(chunksize)


def get_unified_changes(repo_string, commit_hash, filename):
    """
    Returns dataframe with github-like unified diff representation of the content of a file before
    and after a commit for a given git repository, commit hash and filename.
    """
    git_repo = pydriller.GitRepository(repo_string)
    commit = git_repo.get_commit(commit_hash)
    for modification in commit.modifications:
        if modification.filename == filename:
            parsed_lines = git_repo.parse_diff(modification.diff)

            deleted_lines = { x[0]:x[1] for x in parsed_lines['deleted'] }
            added_lines = { x[0]:x[1] for x in parsed_lines['added'] }

            pre_to_post, blocks = identify_edits(deleted_lines, added_lines)

            post_source_code = modification.source_code.split('\n')

            max_line_no = max(max(deleted_lines.keys()),
                              max(added_lines.keys()),
                              len(post_source_code))

            pre = []
            post = []
            action = []
            code = []

            left = 1
            right = 1
            while max(left, right) < max_line_no:
                if left in blocks.keys():
                    cur = left
                    for i in range(blocks[cur][0]):
                        pre.append(left)
                        post.append(None)
                        action.append('-')
                        code.append(deleted_lines[left])
                        left += 1
                    for i in range(blocks[cur][2]):
                        pre.append(None)
                        post.append(right)
                        action.append('+')
                        code.append(added_lines[right])
                        right += 1
                else:
                    if left in pre_to_post.keys():
                        # if left is not in the dictionary nothing has changed
                        if right < pre_to_post[left]:
                            # a block has been added
                            for i in range(pre_to_post[left] - right):
                                pre.append(None)
                                post.append(right)
                                action.append('+')
                                code.append(added_lines[right])
                                right += 1

                    pre.append(left)
                    post.append(right)
                    action.append(None)
                    code.append(post_source_code[right - 1]) # minus one as list starts from 0
                    left += 1
                    right += 1

    return pd.DataFrame({'pre': pre, 'post': post, 'action': action, 'code': code})


def _get_tedges(db_location):
    con = sqlite3.connect(db_location)

    tedges = pd.read_sql("""SELECT x.author_pre as source,
                                   substr(x.pre_commit, 1, 8) as pre_commit,
                                   c_post.author_email AS target,
                                   substr(x.post_commit, 1, 8) AS post_commit,
                                   c_post.committer_date as time,
                                   x.levenshtein_dist as levenshtein_dist
                            FROM (
                                   SELECT c_pre.author_email AS author_pre,
                                          coedits.pre_commit,
                                          coedits.post_commit,
                                          coedits.levenshtein_dist
                                   FROM coedits
                                   JOIN commits AS c_pre
                                   ON substr(c_pre.hash, 1, 8) == coedits.pre_commit) AS x
                                   JOIN commits AS c_post
                                   ON substr(c_post.hash, 1, 8) == substr(x.post_commit, 1, 8
                                 )
                            WHERE source != target""", con)

    tedges.loc[:,'time'] = pd.to_datetime(tedges.time)

    return tedges


def get_coediting_network(db_location, time_from=None, time_to=None):
    tedges = _get_tedges(db_location)

    if time_from == None:
        time_from = min(tedges.time)
    if time_to == None:
        time_to = max(tedges.time)

    t = pp.TemporalNetwork()
    for idx, edge in tedges.iterrows():
        if (edge.time >= time_from) and (edge.time <= time_to):
            t.add_edge(edge.source,
                       edge.target,
                       edge.time.strftime('%Y-%m-%d %H:%M:%S'),
                       directed=True,
                       timestamp_format='%Y-%m-%d %H:%M:%S')
    return t

def _get_bipartite_edges(db_location):
    con = sqlite3.connect(db_location)

    bipartite_edges = pd.read_sql("""SELECT DISTINCT mod_filename AS target,
                                            commits.author_name AS source,
                                            commits.committer_date AS time
                                     FROM coedits
                                     JOIN commits ON coedits.post_commit == commits.hash""", con)

    bipartite_edges.loc[:,'time'] = pd.to_datetime(bipartite_edges.time)

    return bipartite_edges


def get_bipartite_network(db_location, time_from=None, time_to=None):
    bipartite_edges = _get_bipartite_edges(db_location)

    if time_from == None:
        time_from = min(bipartite_edges.time)
    if time_to == None:
        time_to = max(bipartite_edges.time)

    n = pp.Network()
    for idx, edge in bipartite_edges.iterrows():
        if (edge.time >= time_from) and (edge.time <= time_to):
            n.add_edge(edge.source, edge.target)
    return n


def _get_dag_edges(db_location):
    con = sqlite3.connect(db_location)

    dag_edges = pd.read_sql("""SELECT DISTINCT x.author_pre||","||substr(x.pre_commit, 1, 8)
                                        AS source,
                                      c_post.author_email||","|| substr(x.post_commit, 1, 8)
                                        AS target,
                                      c_post.committer_date AS time
                               FROM (
                                      SELECT c_pre.author_email AS author_pre,
                                             coedits.pre_commit,
                                             coedits.post_commit,
                                             coedits.levenshtein_dist
                                      FROM coedits
                                      JOIN commits AS c_pre
                                      ON substr(c_pre.hash, 1, 8) == coedits.pre_commit
                                    ) AS x
                                JOIN (
                                       SELECT *
                                       FROM commits
                                     ) AS c_post
                                ON substr(c_post.hash, 1, 8) == substr(x.post_commit, 1, 8)
                                WHERE x.author_pre != c_post.author_email""", con)

    dag_edges.loc[:,'time'] = pd.to_datetime(dag_edges.time)

    return dag_edges


def get_dag(db_location, time_from=None, time_to=None):
    dag_edges = _get_dag_edges(db_location)

    if time_from == None:
        time_from = min(dag_edges.time)
    if time_to == None:
        time_to = max(dag_edges.time)

    dag = pp.DAG()
    for _, edge in dag_edges.iterrows():
        if (edge.time >= time_from) and (edge.time <= time_to):
            dag.add_edge(edge.source, edge.target)

    dag.topsort()

    return dag


def mine_git_repo(repo_string, sqlite_db_file, exclude=[], no_parallel=False,
                  num_processes=os.cpu_count(), get_block_size=1, use_blocks=True):

    if no_parallel == False:
        process_repo_parallel(repo_string=repo_string, sqlite_db_file=sqlite_db_file,
                              num_processes=num_processes, exclude=exclude,
                              chunksize=get_block_size, use_blocks=use_blocks)
    else:
        df_commits, df_coedits = process_repo_serial(repo_string, exclude, use_blocks=use_blocks)
        con = sqlite3.connect(sqlite_db_file)
        if not df_commits.empty:
            df_commits.to_sql('commits', con, if_exists='append')
        if not df_coedits.empty:
            df_coedits.to_sql('coedits', con, if_exists='append')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extracts commit and co-editing data from git repositories.')
    parser.add_argument('repo', help='Path to repository to be parsed.', type=str)
    parser.add_argument('outfile', help='Path to SQLite DB file storing results.', type=str)
    parser.add_argument('--exclude', help='Exclude path prefixes in given file.', type=str,
        default=None)
    parser.add_argument('--no-parallel', help='Do not use multi-core processing.', dest='parallel',
        action='store_false')
    parser.add_argument('--numprocesses',
        help='Number of CPU cores used for multi-core processing. Defaults to number of CPU cores.',
        default=os.cpu_count(), type=int)
    parser.add_argument('--chunksize', help='Chunk size to be used in multiprocessing.map.',
        default=1, type=int)
    parser.add_argument('--use-blocks',
        help='Compare added and deleted blocks of code rather than lines.', dest='use_blocks',
        action='store_true')
    parser.set_defaults(parallel=True)
    parser.set_defaults(use_blocks=False)

    args = parser.parse_args()
    if args.parallel:
        process_repo_parallel(repo_string=args.repo, sqlite_db_file=args.outfile,
                              num_processes=args.numprocesses, exclude=args.exclude,
                              chunksize=args.chunksize, use_blocks=args.use_blocks)
    else:
        df_commits, df_coedits = process_repo_serial(args.repo, args.exclude,
            use_blocks=args.use_blocks)
        con = sqlite3.connect(args.outfile)
        if not df_commits.empty:
            df_commits.to_sql('commits', con, if_exists='append')
        if not df_coedits.empty:
            df_coedits.to_sql('coedits', con, if_exists='append')