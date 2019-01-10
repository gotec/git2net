#!/usr/bin/python3

import logging
import sqlite3
import os
import argparse

from multiprocessing import Pool
from multiprocessing import Semaphore

import pandas as pd
import progressbar
import numpy as np

import pydriller as pydriller
from pydriller.git_repository import GitCommandError
from Levenshtein import distance as lev_dist

logger = logging.getLogger(__name__)


def chunk_size(lines, k):
    """
    Calculates the size of a chunk of added/deleted lines starting
    in a given line k.

    Parameters
    ----------
    @param lines: dictionary of added or deleted lines
    @param k: line number to check for
    """
    if k not in lines or (k>1 and k-1 in lines):
        chunk = False
        chunksize = 0
    else:
        chunk = True
        chunksize = 1

    while chunk:
        if k+chunksize in lines:
            chunksize += 1
        else:
            chunk = False
    return chunksize

def pre_to_post(deleted_lines, added_lines):
    """
    Maps line numbers between the pre- and post-commit version
    of a modification.
    """
    #print('=======================')
    #print('File: ' + mod.filename)
    #print('=======================')

    #diff = git_repo.parse_diff(mod.diff)
    #print('diff = ' + mod.diff)
    
    #deleted_lines = { }
    #added_lines = { }
    #max_deleted = 0
    #max_added = 0
    #for x in diff['deleted']:
    #    deleted_lines[x[0]] = x[1]
    #    max_deleted = max(x[0], max_deleted)
    #for x in diff['added']:
    #    added_lines[x[0]] = x[1]
    #    max_added = max(x[0], max_added)
    
    # either deleted or added lines must contain items otherwise there would not be a modification to process
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
    left_to_right = {}
    chunks = {} # keys are deleted line numbers, values are tuples containing (number of deleted lines, right line number, number of added lines)
    right = min(min_added, min_deleted)
    left = min(min_added, min_deleted)

    no_right_inc = 0
    both_inc = 0
    jump_right = 0

    while left <= max(max_added, max_deleted):

        # compute size of added and deleted chunks
        chunk_added = chunk_size(added_lines, right)
        chunk_deleted = chunk_size(deleted_lines, left)

        # we ignore pure additions
        if chunk_deleted > 0:
            chunks[left] = (chunk_deleted, right, chunk_added)

        # deleted chunk is larger than added chunk
        if chunk_deleted > chunk_added:
            no_right_inc = chunk_deleted - chunk_added
            both_inc = chunk_added

        # added chunk is larger than deleted chunk
        elif chunk_added > chunk_deleted:
            jump_right = chunk_added - chunk_deleted
            both_inc = chunk_deleted

        # increment left and right counter
        if both_inc>0:
            both_inc -= 1
            left_to_right[left] = right
            left +=1            
            right += 1
        elif no_right_inc>0:
                no_right_inc -= 1
                left_to_right[left] = False                
                left += 1
        else:
            right += jump_right
            left_to_right[left] = right
            left +=1            
            right += 1
            jump_right = 0

    return left_to_right, chunks


def extract_coedits(git_repo, commit, mod, use_blocks=False):
    
    df = pd.DataFrame()

    path = mod.new_path

    diff = mod.diff
    diff = diff.replace('\ No newline at end of file\n', '')

    deleted_lines = { x[0]:x[1] for x in git_repo.parse_diff(diff)['deleted'] }
    added_lines = { x[0]:x[1] for x in git_repo.parse_diff(diff)['added'] }

    left_to_right, chunks = pre_to_post(deleted_lines, added_lines)

    print('use_blocks = ', use_blocks)

    try:
        blame = git_repo.git.blame(commit.hash + '^', '--', path).split('\n')
        if 0=1: #use_blocks:
            for num_line, line in deleted_lines.items():
                if num_line in chunks.keys():
                    chunk = chunks[num_line]

                    blame_fields = blame[num_line - 1].split(' ')
                    buggy_commit_hash = blame_fields[0].replace('^', '')

                    c = {}
                    c['pre_commit'] = [buggy_commit_hash]
                    c['post_commit'] = [commit.hash]
                    c['mod_filename'] = [mod.filename]
                    c['mod_cyclomatic_complexity'] = [mod.complexity]
                    c['mod_loc'] = [mod.nloc]
                    c['mod_old_path'] = [mod.old_path]
                    c['mod_new_path'] = [path]
                    c['mod_token_count'] = [mod.token_count]
                    c['mod_removed'] = [mod.removed]
                    c['mod_added'] = [mod.added]

                    deleted_chunk = []
                    for i in range(num_line, num_line + chunk[0]):
                        deleted_chunk.append(deleted_lines[i])

                    added_chunk = []
                    for i in range(chunk[1], chunk[1] + chunk[2]):
                        added_chunk.append(added_lines[i])

                    print(deleted_chunk)
                    print(added_chunk)
                    

                    #c['pre_line_len'] = [len(line)]
                    #c['pre_line_num'] = [num_line]

                    if left_to_right[num_line] in added_lines:
                        c['post_line_len'] = [len(added_lines[left_to_right[num_line]])]
                        c['post_line_num'] = [left_to_right[num_line]]
                        c['levenshtein_dist'] = [lev_dist(added_lines[left_to_right[num_line]], line)]
                        #print('added line\t\t= ', added_lines[left_to_right[num_line]])
                        #print('levenshtein distance = ', lev_dist(added_lines[left_to_right[num_line]], line))
                    else:
                        #print('removed line')
                        c['post_line_len'] = [0]
                        c['post_line_num'] = [None]
                        c['levenshtein_dist'] = [None]
                    df_t = pd.DataFrame(c)
                    df = pd.concat([df, df_t])
        else:
            for num_line, line in deleted_lines.items():
                blame_fields = blame[num_line - 1].split(' ')
                buggy_commit_hash = blame_fields[0].replace('^', '')
                #buggy_commit = git_repo.get_commit(buggy_commit_hash)
                #print('Previous commit: \t' + buggy_commit_hash)
                #print('Author: \t\t' + buggy_commit.author.email)
                #print('Date: \t\t\t' + buggy_commit.author_date.strftime('%Y-%m-%d %H:%M:%S'))
                #print('blame\t\t\t= ', blame[num_line-1])

                c = {}
                c['pre_commit'] = [buggy_commit_hash]
                c['post_commit'] = [commit.hash]            
                c['pre_line_len'] = [len(line)]
                c['pre_line_num'] = [num_line]
                c['mod_filename'] = [mod.filename]
                c['mod_cyclomatic_complexity'] = [mod.complexity]
                c['mod_loc'] = [mod.nloc]
                c['mod_old_path'] = [mod.old_path]
                c['mod_new_path'] = [path]
                c['mod_token_count'] = [mod.token_count]
                c['mod_removed'] = [mod.removed]
                c['mod_added'] = [mod.added]
                
                #right_to_left = { v:k for k,v in left_to_right.items() if v != False }

                #print('deleted line num\t= ', num_line)
                #print('deleted line\t\t= ', line)
                
                #print('added line num\t\t= ', left_to_right[num_line])
                if left_to_right[num_line] in added_lines:
                    c['post_line_len'] = [len(added_lines[left_to_right[num_line]])]
                    c['post_line_num'] = [left_to_right[num_line]]
                    c['levenshtein_dist'] = [lev_dist(added_lines[left_to_right[num_line]], line)]
                    #print('added line\t\t= ', added_lines[left_to_right[num_line]])
                    #print('levenshtein distance = ', lev_dist(added_lines[left_to_right[num_line]], line))
                else:
                    #print('removed line')
                    c['post_line_len'] = [0]
                    c['post_line_num'] = [None]
                    c['levenshtein_dist'] = [None]
                df_t = pd.DataFrame(c)
                df = pd.concat([df, df_t])

    except GitCommandError:
        logger.debug("Could not found file %s in commit %s. Probably a double rename!", mod.filename,
                        commit.hash)
    
    return df


def process_commit(git_repo, commit, exclude_paths = set(), use_blocks = False):
    df_commit = pd.DataFrame()
    df_coedits = pd.DataFrame()
    #print(commit.hash)

    # parse commit
    c = {}
    c['hash'] = [commit.hash]
    c['author_email'] = [commit.author.email]
    c['author_name'] = [commit.author.name]
    c['committer_email'] = [commit.committer.email]
    c['committer_name'] = [commit.committer.name]
    c['author_date'] = [commit.author_date.strftime('%Y-%m-%d %H:%M:%S')]
    c['committer_date'] = [commit.committer_date.strftime('%Y-%m-%d %H:%M:%S')]
    c['committer_timezone'] = [commit.committer_timezone]
    c['modifications'] = [len(commit.modifications)]
    c['msg_len'] = [len(commit.msg)]
    c['project_name'] = [commit.project_name]
    c['parents'] = [ ','.join(commit.parents)]
    c['merge'] = [commit.merge]
    c['in_main_branch'] = [commit.in_main_branch]
    c['branches'] = [ ','.join(commit.branches)]
    
    df_commit = pd.DataFrame(c)
    
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
        #else:
        #    print('skipping file {0} in commit {1}'.format(excluded_path, commit.hash))
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
    with progressbar.ProgressBar(max_value=len(commits)) as bar:
        for commit in commits:
            df_1, df_2 = process_commit(git_repo, commit, exclude_paths, use_blocks=use_blocks)
            df_commits = pd.concat([df_commits, df_1])
            df_coedits = pd.concat([df_coedits, df_2])
            i += 1
            bar.update(i)
    return df_commits, df_coedits

semaphore = Semaphore(1)


def process_commit_parallel(arg):
    repo_string, commit_hash, sqlite_db_file, exclude, use_blocks = arg
    git_repo = pydriller.GitRepository(repo_string)

    exclude_paths = []
    if exclude:
        with open(exclude) as f:
            exclude_paths = [x.strip() for x in f.readlines()]

    df_1, df_2 = process_commit(git_repo, git_repo.get_commit(commit_hash), exclude_paths, use_blocks=use_blocks)    
    
    with semaphore:
        con = sqlite3.connect(sqlite_db_file)
        if not df_1.empty:
            df_1.to_sql('commits', con, if_exists='append')
        if not df_2.empty:
            df_2.to_sql('coedits', con, if_exists='append')
    return True

def process_repo_parallel(repo_string, sqlite_db_file, num_processes=os.cpu_count(), exclude=None, chunksize=1):
    
    print('Using {0} workers.'.format(num_processes))    
    git_repo = pydriller.GitRepository(repo_string)
    args = [ (repo_string, c.hash, sqlite_db_file, exclude) for c in git_repo.get_list_commits()]

    p = Pool(num_processes)
    with progressbar.ProgressBar(max_value=len(args)) as bar:
        for i,_ in enumerate(p.imap_unordered(process_commit_parallel, args, chunksize=chunksize)):
            bar.update(i)


parser = argparse.ArgumentParser(description='Extracts commit and co-editing data from git repositories.')
parser.add_argument('repo', help='path to repository to be parsed.', type=str)
parser.add_argument('outfile', help='path to SQLite DB file storing results.', type=str)
#parser.add_argument('--parallel', help='use multi-core processing. Default.', dest='parallel', action='store_true')
parser.add_argument('--exclude', help='exclude path prefixes in given file', type=str, default=None)
parser.add_argument('--no-parallel', help='do not use multi-core processing.', dest='parallel', action='store_false')
parser.add_argument('--numprocesses', help='number of CPU cores to use for multi-core processing. Defaults to number of CPU cores.', default=os.cpu_count(), type=int)
parser.add_argument('--chunksize', help='chunk size to be used in multiprocessing.map.', default=1, type=int)
parser.add_argument('--use-blocks', help='compare added and deleted blocks of code rather than lines', dest='use_blocks', action='store_true')
parser.set_defaults(parallel=True)
parser.set_defaults(use_blocks=False)

args = parser.parse_args()
if args.parallel:
    process_repo_parallel(repo_string=args.repo, sqlite_db_file=args.outfile, num_processes=args.numprocesses, exclude=args.exclude, chunksize=args.chunksize)
else:
    df_commits, df_coedits = process_repo_serial(args.repo, args.exclude, use_blocks=args.use_blocks)
    con = sqlite3.connect(args.outfile)
    if not df_commits.empty:
        df_commits.to_sql('commits', con, if_exists='append')
    if not df_coedits.empty:
        df_coedits.to_sql('coedits', con, if_exists='append')