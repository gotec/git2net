import git2net
import pathpy as pp
import pytest
import pydriller
import numpy as np
import lizard
import os
from datetime import datetime
import sys
import pandas as pd
import sqlite3
import time
import shutil
import git

@pytest.fixture(scope="function")
def git_repo_dir():
    yield 'test_repos/test_repo_1'

@pytest.fixture(scope="function")
def github_repo_dir():
    repo_dir = 'test_repos/git2net_test'
    yield repo_dir
    shutil.rmtree(repo_dir)
    
@pytest.fixture(scope="function")
def github_url_short():
    yield 'gotec/git2net'
    
@pytest.fixture(scope="function")
def github_url_full():
    yield 'https://github.com/gotec/git2net'

@pytest.fixture(scope="function")
def github_url_invalid():
    yield 'invalid'

@pytest.fixture(scope="function")
def sqlite_db_file():
    db_path = 'tests/test_repo_1.db'
    yield db_path
    if os.path.exists(db_path):
        os.remove(db_path)



    
def test_get_commit_dag(git_repo_dir):
    print('rootdir', [f for f in os.listdir('.')])
    print('rootdir-1', [f for f in os.listdir('test_repos')])
    print('test_repos', [f for f in os.listdir('test_repos/test_repo_1')])
    
    dag = git2net.get_commit_dag(git_repo_dir)
    expected_edges = [('e4448e8', 'f343ed5'), ('f343ed5', '6b531fc'), ('6b531fc', 'b17c2c3'),
                      ('b17c2c3', '2b00f48'), ('2b00f48', '59da499'), ('2b00f48', 'b21583e'),
                      ('b21583e', '7d140b9'), ('59da499', '7d140b9'), ('7d140b9', '9e28e38'),
                      ('7d140b9', '16bbc87'), ('9e28e38', '080220d'), ('16bbc87', '080220d'),
                      ('16bbc87', 'eadd9d4'), ('080220d', '02b4a6f'), ('02b4a6f', '9798c44'),
                      ('eadd9d4', '9798c44'), ('9798c44', '2f4f139'), ('9798c44', '2ce5105'),
                      ('2ce5105', '9602507'), ('2f4f139', '9602507'), ('9602507', '2edf3d9'),
                      ('9602507', '2c3300a'), ('9602507', '00f3bbe'), ('2c3300a', '4b5f698'),
                      ('2edf3d9', '4b5f698'), ('00f3bbe', 'deff6c8'), ('2edf3d9', 'deff6c8'),
                      ('4b5f698', 'dcf060d'), ('deff6c8', 'dcf060d'), ('dcf060d', '5606e82'),
                      ('5606e82', '1adc153'), ('5606e82', 'e8be9c6'), ('e8be9c6', '97b5e43'),
                      ('1adc153', '97b5e43'), ('97b5e43', '1c038ed'), ('1c038ed', '94a9da2'),
                      ('94a9da2', '83214bc'), ('83214bc', '82351f1'), ('83214bc', 'e6172fd'),
                      ('e6172fd', 'a1b3307'), ('82351f1', 'a1b3307'), ('a1b3307', '7df20cb'),
                      ('7df20cb', 'be2c55b')]
    dag.topsort()
    
    assert list(dag.edges.keys()) == expected_edges
    assert dag.is_acyclic


def test_extract_edits_1(git_repo_dir):
    commit_hash = 'b17c2c321ce8d299de3d063ca0a1b0b363477505'
    filename = 'first_lines.txt'

    extraction_settings = {'use_blocks': False,
                           'blame_options': ['-C', '-C', '-C4', '--show-number', '--line-porcelain'],
                           'extract_complexity': True,
                           'extract_text': True}
    git_repo = pydriller.Git(git_repo_dir)
    commit = git_repo.get_commit(commit_hash)
    for mod in commit.modified_files:
        if mod.filename == filename:
            df = git2net.extraction._extract_edits(git_repo, commit, mod, extraction_settings)
            
    assert len(df) == 3
    assert df.at[0, 'original_commit_addition'] == 'e4448e87541d19d139b9d033b2578941a53d1f97'
    assert df.at[1, 'original_commit_addition'] == '6b531fcb57d5b9d98dd983cb65357d82ccca647b'
    assert df.at[2, 'original_commit_addition'] == None # as there is no match due to line ending
    # to obtain match, the -w option is required in git blame, however this leads to wrong matches
    # with lines that were not copied.
    
    
def test_extract_edits_2(git_repo_dir):
    commit_hash = 'b17c2c321ce8d299de3d063ca0a1b0b363477505'
    filename = 'first_lines.txt'

    extraction_settings = {'use_blocks': True,
                           'blame_options': ['-C', '-C', '-C4', '--show-number', '--line-porcelain'],
                           'extract_complexity': True,
                           'extract_text': True}
    
    git_repo = pydriller.Git(git_repo_dir)
    commit = git_repo.get_commit(commit_hash)
    df = None
    for mod in commit.modified_files:
        if mod.filename == filename:
            df = git2net.extraction._extract_edits(git_repo, commit, mod, extraction_settings)
    assert len(df) == 1
    assert df.at[0, 'original_commit_addition'] == 'not available with use_blocks'


def test_identify_edits(git_repo_dir):
    commit_hash = 'f343ed53ee64717f85135c4b8d3f6bd018be80ad'
    filename = 'text_file.txt'

    extraction_settings = {'use_blocks': False}
    
    git_repo = pydriller.Git(git_repo_dir)
    commit = git_repo.get_commit(commit_hash)
    for x in commit.modified_files:
        if x.filename == filename:
            mod = x

    parsed_lines = mod.diff_parsed

    deleted_lines = { x[0]:x[1] for x in parsed_lines['deleted'] }
    added_lines = { x[0]:x[1] for x in parsed_lines['added'] }

    _, edits = git2net.extraction._identify_edits(deleted_lines, added_lines, extraction_settings)
    assert list(edits.type) == ['deletion', 'replacement', 'deletion', 'replacement', 'addition',
                                'addition', 'addition']

def test_process_commit(git_repo_dir):
    commit_hash = 'f343ed53ee64717f85135c4b8d3f6bd018be80ad'
    
    extraction_settings = {'use_blocks': False,
                           'exclude': [],
                           'blame_options': ['-C', '--show-number', '--line-porcelain'],
                           'timeout': 0,
                           'max_modifications': 0,
                           'no_of_processes': 4,
                           'extract_text': True,
                           'extract_complexity': True,
                           'timeout': 0}
    
    args = {'git_repo_dir': git_repo_dir, 'commit_hash': commit_hash,
            'extraction_settings': extraction_settings}

    res_dict, _, _ = git2net.extraction._process_commit(args)
    assert list(res_dict.keys()) == ['commit', 'edits']


def test_get_unified_changes(git_repo_dir):
    commit_hash = 'e8be9c6abe76c809a567866e411350e76eb45e49'
    filename = 'text_file.txt'
    unified_changes = git2net.get_unified_changes(git_repo_dir, commit_hash, filename)
    expected_code = ['A0', 'B1', 'B2', 'B3', 'A1', 'C2', 'C3', 'C4', 'B2', 'B3', 'B4', 'A5', 'A6',
                     'A7', 'F8', 'F9', 'F10', 'F11', 'F12', 'B8', 'B9', 'B10', 'B11', 'B12']
    assert list(unified_changes.code) == expected_code


def test_mine_git_repo_sequential(git_repo_dir, sqlite_db_file):
    git2net.mine_git_repo(git_repo_dir, sqlite_db_file, blame_C='CCC4', no_of_processes=1)
    assert git2net.check_mining_complete(git_repo_dir, sqlite_db_file)


def test_mine_git_repo(git_repo_dir, sqlite_db_file):
    git2net.mine_git_repo(git_repo_dir, sqlite_db_file, blame_C='CCC4', extract_merge_deletions=True,
                          extract_merges=True, extract_text=True)
    assert git2net.check_mining_complete(git_repo_dir, sqlite_db_file)

def test_disambiguation(git_repo_dir, sqlite_db_file):
    git2net.mine_git_repo(git_repo_dir, sqlite_db_file)
    git2net.disambiguate_aliases_db(sqlite_db_file)
    
    with sqlite3.connect(sqlite_db_file) as con:
        author_id = pd.read_sql('SELECT author_id FROM commits', con)
    
    assert not author_id.empty

def test_get_line_editing_paths_1(sqlite_db_file, git_repo_dir):
    # No database exists
    with pytest.raises(Exception) as e:
        _ = git2net.get_line_editing_paths(sqlite_db_file, git_repo_dir, author_identifier='author_name')
    assert e.value.args[0].startswith('You either provided no database')

        
def test_get_line_editing_paths_2(sqlite_db_file, git_repo_dir):
    # An empty database exists
    with sqlite3.connect(sqlite_db_file) as con:
        pd.DataFrame().to_sql('git2net', con)
    with pytest.raises(Exception) as e:
        _ = git2net.get_line_editing_paths(sqlite_db_file, git_repo_dir, author_identifier='author_name')
    assert e.value.args[0].startswith('You either provided no database')
    
    
def test_get_line_editing_paths_3(sqlite_db_file, git_repo_dir):
    git2net.mine_git_repo(git_repo_dir, sqlite_db_file, use_blocks=True)
    
    with sqlite3.connect(sqlite_db_file) as con:
        pd.DataFrame().to_sql('git2net', con)
    with pytest.raises(Exception) as e:
        _ = git2net.get_line_editing_paths(sqlite_db_file, git_repo_dir, author_identifier='author_name')
    assert e.value.args[0].startswith('Invalid database. A database mined with')
    
    
def test_get_line_editing_paths_4(sqlite_db_file, git_repo_dir):
    git2net.mine_git_repo(git_repo_dir, sqlite_db_file)
    git2net.disambiguate_aliases_db(sqlite_db_file)
    
    paths, dag, node_info, edge_info = git2net.get_line_editing_paths(sqlite_db_file, git_repo_dir,
                                                                      with_start=True)
    
    assert len(dag.isolate_nodes()) == 0
    
    _ = git2net.get_line_editing_paths(sqlite_db_file, git_repo_dir,
                                       with_start=False,
                                       commit_hashes= ['e4448e87541d19d139b9d033b2578941a53d1f97',
                                                       'f343ed53ee64717f85135c4b8d3f6bd018be80ad', 
                                                       '6b531fcb57d5b9d98dd983cb65357d82ccca647b', 
                                                       'b17c2c321ce8d299de3d063ca0a1b0b363477505', 
                                                       '2b00f48ff42cf5c12646cb0553e5481b49bd78f7'],
                                       author_identifier='author_name')
    _ = git2net.get_line_editing_paths(sqlite_db_file, git_repo_dir,
                                       with_start=True, merge_renaming=True,
                                       file_paths=['text_file.txt'],
                                       author_identifier='author_email')
    
    with pytest.raises(Exception) as e:
        _ = git2net.get_line_editing_paths(sqlite_db_file, git_repo_dir,
                                           with_start=True,
                                           author_identifier='invalid')
    assert e.value.args[0].startswith('author_identifier must be from')
    
    


def test_get_commit_editing_dag_1(sqlite_db_file, git_repo_dir):
    git2net.mine_git_repo(git_repo_dir, sqlite_db_file, blame_C='CCC4', extract_merge_deletions=True,
                          extract_merges=True, extract_text=True)
    
    dag, node_info, edge_info = git2net.get_commit_editing_dag(sqlite_db_file)

    assert len(dag.isolate_nodes()) == 0
    assert len(dag.nodes) == 36
    assert len(dag.successors[None]) == 12


def test_get_commit_editing_dag_2(sqlite_db_file, git_repo_dir):
    git2net.mine_git_repo(git_repo_dir, sqlite_db_file, blame_C='CCC4', extract_merge_deletions=True,
                          extract_merges=True, extract_text=True)
    
    time_from = datetime(2019, 2, 12, 11, 0, 0)
    time_to = datetime(2019, 2, 12, 12, 0, 0)

    dag, node_info, edge_info = git2net.get_commit_editing_dag(sqlite_db_file,
                                                               time_from=time_from,
                                                               time_to=time_to)

    assert len(dag.isolate_nodes()) == 0
    assert len(dag.nodes) == 15
    assert len(dag.successors[None]) == 6


def test_get_commit_editing_dag_3(sqlite_db_file, git_repo_dir):
    git2net.mine_git_repo(git_repo_dir, sqlite_db_file, blame_C='CCC4', extract_merge_deletions=True,
                          extract_merges=True, extract_text=True)
    
    time_from = datetime(2019, 2, 12, 12, 0, 0)
    time_to = datetime(2019, 2, 12, 13, 0, 0)
    filename = 'text_file.txt'

    dag, node_info, edge_info = git2net.get_commit_editing_dag(sqlite_db_file,
                                                               time_from=time_from,
                                                               time_to=time_to,
                                                               filename=filename)

    assert len(dag.isolate_nodes()) == 0
    assert len(dag.nodes) == 17
    assert len(dag.successors[None]) == 1


def test_get_coediting_network(sqlite_db_file, git_repo_dir):
    git2net.mine_git_repo(git_repo_dir, sqlite_db_file)
    git2net.disambiguate_aliases_db(sqlite_db_file)
    
    _ = git2net.get_coediting_network(sqlite_db_file)
    
    time_from = datetime(2019, 2, 12, 11, 0, 0)
    time_to = datetime(2019, 2, 12, 11, 15, 0)

    t, node_info, edge_info = git2net.get_coediting_network(sqlite_db_file, time_from=time_from,
                                                            time_to=time_to)

    expected_edges = [('Author B', 'Author A', 1549965657),
                      ('Author A', 'Author B', 1549966134),
                      ('Author B', 'Author A', 1549966184),
                      ('Author C', 'Author B', 1549966309),
                      ('Author C', 'Author A', 1549966309),
                      ('Author C', 'Author A', 1549966309),
                      ('Author B', 'Author A', 1549966356),
                      ('Author B', 'Author A', 1549965738),
                      ('Author C', 'Author A', 1549966451),
                      ('Author C', 'Author A', 1549966451),
                      ('Author C', 'Author A', 1549966451)]

    assert len(set(t.tedges).difference(set(expected_edges))) == 0

    t, node_info, edge_info = git2net.get_coediting_network(sqlite_db_file, time_from=time_from,
                                                            time_to=time_to,
                                                            author_identifier='author_name')
    t, node_info, edge_info = git2net.get_coediting_network(sqlite_db_file, time_from=time_from,
                                                            time_to=time_to,
                                                            author_identifier='author_email')
    
    with pytest.raises(Exception) as e:
        t, node_info, edge_info = git2net.get_coediting_network(sqlite_db_file, time_from=time_from,
                                                                time_to=time_to,
                                                                author_identifier='invalid')
    assert e.value.args[0].startswith('author_identifier must be from')
    
    
    
    

def test_get_coauthorship_network(sqlite_db_file, git_repo_dir):
    git2net.mine_git_repo(git_repo_dir, sqlite_db_file)
    git2net.disambiguate_aliases_db(sqlite_db_file)
    
    time_from = datetime(2019, 2, 12, 12, 0, 0)
    time_to = datetime(2019, 2, 12, 12, 15, 0)

    n, node_info, edge_info = git2net.get_coauthorship_network(sqlite_db_file, time_from=time_from,
                                                               time_to=time_to)

    expected_nonzero_rows = [0, 0, 1, 1, 2, 2]
    expected_nonzero_columns = [1, 2, 0, 2, 0, 1]

    assert list(n.adjacency_matrix().nonzero()[0]) == expected_nonzero_rows
    assert list(n.adjacency_matrix().nonzero()[1]) == expected_nonzero_columns
    
    n, node_info, edge_info = git2net.get_coauthorship_network(sqlite_db_file, time_from=time_from,
                                                               time_to=time_to,
                                                               author_identifier='author_name')
    n, node_info, edge_info = git2net.get_coauthorship_network(sqlite_db_file, time_from=time_from,
                                                               time_to=time_to,
                                                               author_identifier='author_email')
    
    with pytest.raises(Exception) as e:
        n, node_info, edge_info = git2net.get_coauthorship_network(sqlite_db_file, time_from=time_from,
                                                                   time_to=time_to,
                                                                   author_identifier='invalid')
    assert e.value.args[0].startswith('author_identifier must be from')


def test_get_bipartite_network(sqlite_db_file, git_repo_dir):
    git2net.mine_git_repo(git_repo_dir, sqlite_db_file)
    git2net.disambiguate_aliases_db(sqlite_db_file)
    
    time_from = datetime(2019, 2, 12, 11, 0, 0)
    time_to = datetime(2019, 2, 12, 11, 10, 0)

    t, node_info, edge_info = git2net.get_bipartite_network(sqlite_db_file, time_from=time_from,
                                                            time_to=time_to)

    expected_edges = [('Author A', 'text_file.txt', 1549965641),
    ('Author B', 'text_file.txt', 1549965657),
    ('Author A', 'text_file.txt', 1549966134),
    ('Author B', 'text_file.txt', 1549966184),
    ('Author B', 'text_file.txt', 1549965738)]

    assert len(set(t.tedges).difference(set(expected_edges))) == 0

    t, node_info, edge_info = git2net.get_bipartite_network(sqlite_db_file, time_from=time_from,
                                                            time_to=time_to,
                                                            author_identifier='author_name')
    t, node_info, edge_info = git2net.get_bipartite_network(sqlite_db_file, time_from=time_from,
                                                            time_to=time_to,
                                                            author_identifier='author_email')
    
    with pytest.raises(Exception) as e:
        t, node_info, edge_info = git2net.get_bipartite_network(sqlite_db_file, time_from=time_from,
                                                                time_to=time_to,
                                                                author_identifier='invalid')
    assert e.value.args[0].startswith('author_identifier must be from')
        
    
def test_process_commit_merge(git_repo_dir):
    commit_hash = 'dcf060d5aa93077c84552ce6ed56a0f0a37e4dca'
    
    extraction_settings = {'use_blocks': False,
                           'exclude': [],
                           'blame_options': ['-C', '--show-number', '--line-porcelain'],
                           'timeout': 0,
                           'max_modifications': 0,
                           'no_of_processes': 4,
                           'extract_text': True,
                           'extract_merges': True,
                           'extract_complexity': True,
                           'extract_merge_deletions': True}
    
    args = {'git_repo_dir': git_repo_dir, 'commit_hash': commit_hash,
            'extraction_settings': extraction_settings}

    res_dict, _, _ = git2net.extraction._process_commit(args)

    assert list(res_dict['edits']['edit_type']) == ['deletion']*7
    assert list(res_dict['edits']['pre_starting_line_no']) == [6,7,8,12,13,1,2]


def test_process_commit_merge2(git_repo_dir):
    commit_hash = '96025072a3e1b2f466ef56053bbdf4c9c0e927f0'
    
    extraction_settings = {'use_blocks': False,
                           'exclude': [],
                           'blame_options': ['-C', '--show-number', '--line-porcelain'],
                           'timeout': 0,
                           'max_modifications': 0,
                           'no_of_processes': 4,
                           'extract_text': False,
                           'extract_merges': True,
                           'extract_complexity': True,
                           'extract_merge_deletions': True}
    
    args = {'git_repo_dir': git_repo_dir, 'commit_hash': commit_hash,
            'extraction_settings': extraction_settings}

    res_dict, _, _ = git2net.extraction._process_commit(args)

    assert list(res_dict['edits']['edit_type']) == ['replacement']*6
    assert list(res_dict['edits']['pre_starting_line_no']) == [1,2,3,1,2,3]
    
    
def test_compute_halstead_effort():
    filename = 'test.c'
    source_code = """
                  main()
                  {
                      int a, b, c, avg;
                      scanf("%d %d %d", &a, &b, &c);
                      avg = (a+b+c)/3;
                      printf("avg = %d", avg);
                  }
                  """
    
    unknown_filename = 'test.git2net'
    
    HE = git2net.complexity._compute_halstead_effort(filename, source_code)
    HE_unknown = git2net.complexity._compute_halstead_effort(unknown_filename, source_code)
    assert HE == 2196.1587113893806 # obtained from run of multimetric
    assert HE_unknown == 0
    
    
def test_compute_complexity_measures(git_repo_dir):
    # Addition
    args = {'git_repo_dir': '../git2net/test_repos/test_repo_1/',
            'commit_hash': 'e4448e87541d19d139b9d033b2578941a53d1f97',
            'old_path': 'None',
            'new_path': 'text_file.txt',
            'events': -1,
            'levenshtein_distance': -1}

    res = git2net.complexity._compute_complexity_measures(args)

    assert res.old_path.values[0] == args['old_path']
    assert res.new_path.values[0] == args['new_path']
    assert res.events.values[0] == args['events']
    assert res.levenshtein_distance.values[0] == args['levenshtein_distance']
    assert res.HE_pre.values[0] == 0
    assert res.CCN_pre.values[0] == 0
    assert res.NLOC_pre.values[0] == 0
    assert res.TOK_pre.values[0] == 0
    assert res.FUN_pre.values[0] == 0
    assert res.HE_post.values[0] == 1
    assert res.CCN_post.values[0] == 0
    assert res.NLOC_post.values[0] == 10
    assert res.TOK_post.values[0] == 10
    assert res.FUN_post.values[0] == 0
    assert res.HE_delta.values[0] == 1
    assert res.CCN_delta.values[0] == 0
    assert res.NLOC_delta.values[0] == 10
    assert res.TOK_delta.values[0] == 10
    assert res.FUN_delta.values[0] == 0
    
    # Replacement
    args = {'git_repo_dir': '../git2net/test_repos/test_repo_1/',
            'commit_hash': 'b17c2c321ce8d299de3d063ca0a1b0b363477505',
            'old_path': 'text_file.txt',
            'new_path': 'text_file.txt',
            'events': -1,
            'levenshtein_distance': -1}
    
    res = git2net.complexity._compute_complexity_measures(args)

    assert res.old_path.values[0] == args['old_path']
    assert res.new_path.values[0] == args['new_path']
    assert res.events.values[0] == args['events']
    assert res.levenshtein_distance.values[0] == args['levenshtein_distance']
    assert res.HE_pre.values[0] == 1
    assert res.CCN_pre.values[0] == 0
    assert res.NLOC_pre.values[0] == 10
    assert res.TOK_pre.values[0] == 10
    assert res.FUN_pre.values[0] == 0
    assert res.HE_post.values[0] == 1
    assert res.CCN_post.values[0] == 0
    assert res.NLOC_post.values[0] == 4
    assert res.TOK_post.values[0] == 4
    assert res.FUN_post.values[0] == 0
    assert res.HE_delta.values[0] == 0
    assert res.CCN_delta.values[0] == 0
    assert res.NLOC_delta.values[0] == -6
    assert res.TOK_delta.values[0] == -6
    assert res.FUN_delta.values[0] == 0
    
    
    # Deletion
    args = {'git_repo_dir': '../git2net/test_repos/test_repo_1/',
            'commit_hash': '7df20cba2ba34e1e316d19a23e4f573e1007ec75',
            'old_path': 'test_file_2.txt',
            'new_path': 'None',
            'events': -1,
            'levenshtein_distance': -1}
    
    res = git2net.complexity._compute_complexity_measures(args)

    assert res.old_path.values[0] == args['old_path']
    assert res.new_path.values[0] == args['new_path']
    assert res.events.values[0] == args['events']
    assert res.levenshtein_distance.values[0] == args['levenshtein_distance']
    assert res.HE_pre.values[0] == 1
    assert res.CCN_pre.values[0] == 0
    assert res.NLOC_pre.values[0] == 10
    assert res.TOK_pre.values[0] == 10
    assert res.FUN_pre.values[0] == 0
    assert res.HE_post.values[0] == 0
    assert res.CCN_post.values[0] == 0
    assert res.NLOC_post.values[0] == 0
    assert res.TOK_post.values[0] == 0
    assert res.FUN_post.values[0] == 0
    assert res.HE_delta.values[0] == -1
    assert res.CCN_delta.values[0] == 0
    assert res.NLOC_delta.values[0] == -10
    assert res.TOK_delta.values[0] == -10
    assert res.FUN_delta.values[0] == 0
    
    
def test_compute_complexity(git_repo_dir, sqlite_db_file):
    git2net.mine_git_repo(git_repo_dir, sqlite_db_file)
    
    # compute the complexity for everything in the database
    git2net.compute_complexity(git_repo_dir, sqlite_db_file, read_chunksize = 1, write_chunksize = 20)
    
    with sqlite3.connect(sqlite_db_file) as con:
        complexity = pd.read_sql('SELECT * FROM complexity', con)
    
    assert not complexity.empty
    
    # Repeat the computation. Now nothing should happen.
    git2net.compute_complexity(git_repo_dir, sqlite_db_file)
        
    with sqlite3.connect(sqlite_db_file) as con:
        complexity2 = pd.read_sql('SELECT * FROM complexity', con)
        
    assert complexity.equals(complexity2)
    
    
def test_Timeout():
    finished = False
    with git2net.extraction.Timeout(2) as timeout:
        time.sleep(10)
        finished = True
    assert not finished
    assert timeout.timed_out
    
    finished = False
    with git2net.extraction.Timeout(2) as timeout:
        time.sleep(1)
        finished = True
    assert finished
    assert not timeout.timed_out
    
    
def test_mine_github(github_url_short, github_repo_dir, sqlite_db_file):
    git2net.mine_github(github_url_short, github_repo_dir, sqlite_db_file)
    assert git2net.check_mining_complete(github_repo_dir, sqlite_db_file)

    
def test_mine_github_2(github_url_full, github_repo_dir, sqlite_db_file):
    git2net.mine_github(github_url_full, github_repo_dir, sqlite_db_file, branch='object-oriented', use_blocks=True)
    assert git2net.check_mining_complete(github_repo_dir, sqlite_db_file)
    
    
def test_mine_github_3(github_url_invalid):
    github_repo_dir = 'invalid'
    sqlite_db_file = 'invalid'
    
    with pytest.raises(Exception) as e:
        git2net.mine_github(github_url_invalid, github_repo_dir, sqlite_db_file)
    assert e.value.args[0].startswith('Invalid github_url provided.')

        
def test_mine_git_repo_resume(github_url_full, github_repo_dir, sqlite_db_file):
    local_directory = '/'.join(github_repo_dir.split('/')[:-1])
    git_repo_folder = github_repo_dir.split('/')[-1]
    
    git.Git(local_directory).clone(github_url_full, git_repo_folder)
    
    with pytest.raises(Exception) as e:
        git2net.mining_state_summary(github_repo_dir, sqlite_db_file, all_branches=True)
    assert e.value.args[0].startswith('Found no database at provided path.')
        
    with pytest.raises(Exception) as e:
        git2net.check_mining_complete(github_repo_dir, sqlite_db_file)
    assert e.value.args[0].startswith('Found no database at provided path.')
        
    git_repo = pydriller.Git(github_repo_dir)
    commits = [c.hash for c in git_repo.get_list_commits()]
    
    git2net.mine_git_repo(github_repo_dir, sqlite_db_file, commits=commits[:10])

    with pytest.raises(Exception) as e:
        git2net.visualisation._ensure_author_id_exists(sqlite_db_file)
    assert e.value.args[0].startswith('The author_id is not yet computed.')
    
    assert not git2net.check_mining_complete(github_repo_dir, sqlite_db_file)
    
    u_commits_info = git2net.mining_state_summary(github_repo_dir, sqlite_db_file, all_branches=True)
    complete, missing = git2net.check_mining_complete(github_repo_dir, sqlite_db_file, return_number_missing=True)
    
    assert len(u_commits_info) == (len(commits) - 10)       
    assert missing == (len(commits) - 10) 
    
    git2net.disambiguate_aliases_db(sqlite_db_file)
        
    git2net.mine_git_repo(github_repo_dir, sqlite_db_file)
    assert git2net.check_mining_complete(github_repo_dir, sqlite_db_file)
    
    with pytest.raises(Exception) as e:
        git2net.visualisation._ensure_author_id_exists(sqlite_db_file)
    assert e.value.args[0].startswith('The author_id is missing entries.')
    
    
def test_mine_git_repo_exceptions_1(git_repo_dir, sqlite_db_file):
    with pytest.raises(Exception) as e:
        git2net.check_mining_complete(git_repo_dir, sqlite_db_file)
    assert e.value.args[0] == 'Found no database at provided path.'

    with open(sqlite_db_file, 'w') as f:
        f.write('Hello World!')

    with pytest.raises(Exception) as e:
        git2net.check_mining_complete(git_repo_dir, sqlite_db_file)
    assert e.value.args[0] == 'The provided file is not a compatible database.'

    assert os.path.exists(sqlite_db_file)
    os.remove(sqlite_db_file)
    with sqlite3.connect(sqlite_db_file) as con:
        pd.DataFrame().to_sql('git2net', con)

    assert not git2net.check_mining_complete(git_repo_dir, sqlite_db_file)

    with pytest.raises(Exception) as e:
        git2net.mine_git_repo(git_repo_dir, sqlite_db_file)
    assert e.value.args[0].startswith('Found a database on provided path that was likely not created')
        
        
def test_mine_git_repo_exceptions_2(git_repo_dir, sqlite_db_file):
    git_repo = pydriller.Git(git_repo_dir)
    commits = [c.hash for c in git_repo.get_list_commits()]

    git2net.mine_git_repo(git_repo_dir, sqlite_db_file, commits=commits[:1])

    with pytest.raises(Exception) as e:
        git2net.mine_git_repo(git_repo_dir, sqlite_db_file, use_blocks=True)
    assert e.value.args[0].startswith('Found a database on provided path that was created')
        
    
def test_mine_git_repo_exceptions_3(git_repo_dir, github_url_short, github_repo_dir, sqlite_db_file):
    git2net.mine_github(github_url_short, github_repo_dir, sqlite_db_file)

    with pytest.raises(Exception) as e:
        git2net.mining_state_summary(git_repo_dir, sqlite_db_file)
    assert e.value.args[0].startswith('The database does not match the provided repository.')
        
    with pytest.raises(Exception) as e:
        git2net.mine_git_repo(git_repo_dir, sqlite_db_file)
    assert e.value.args[0].startswith('Found a database that was created with identical settings. However')
    
def test_get_edited_file_paths_since_split(git_repo_dir):
    
    git_repo = pydriller.Git(git_repo_dir)
    commit = git_repo.get_commit('a1b3307e5066455508447e5d2811cc1eacb10a0c')
    
    edited_file_paths = git2net.extraction._get_edited_file_paths_since_split(git_repo, commit)
    
    assert edited_file_paths == {'test_file_2.txt', 'test_file_3.txt', 'test_folder/text_file_changed_name.txt'}
    

def test_identify_file_renaming(git_repo_dir):
    
    dag, aliases = git2net.identify_file_renaming(git_repo_dir)
    
    expected_aliases = {'test_file_2.txt': 'text_file.txt',
                        'test_folder/text_file_changed_name.txt': 'text_file.txt',
                        'text_file_changed_name.txt': 'text_file.txt'}
    
    expected_edges = [('text_file_changed_name.txt', 'text_file.txt'), 
                      ('test_folder/text_file_changed_name.txt', 'text_file_changed_name.txt'),
                      ('test_file_2.txt', 'test_folder/text_file_changed_name.txt')]
    
    assert aliases == expected_aliases
    
    assert len(dag.nodes) == 8
    assert len(dag.roots) == 5
    assert len(dag.leafs) == 5
    assert list(dag.edges.keys()) == expected_edges

    
def test_visualisation_github(github_url_short, github_repo_dir, sqlite_db_file):
    git2net.mine_github(github_url_short, github_repo_dir, sqlite_db_file)
    assert git2net.check_mining_complete(github_repo_dir, sqlite_db_file)
    
    git2net.disambiguate_aliases_db(sqlite_db_file)
    assert git2net.visualisation._ensure_author_id_exists(sqlite_db_file)
    
    # Assures no visualisation function throws error on real data
    paths, dag, node_info, edge_info = git2net.get_line_editing_paths(sqlite_db_file, github_repo_dir)
    assert type(paths) == pp.Paths
    assert type(dag) == pp.DAG
    assert type(node_info) == dict
    assert type(edge_info) == dict
    
    dag, node_info, edge_info = git2net.get_commit_editing_dag(sqlite_db_file)
    assert type(dag) == pp.DAG
    assert type(node_info) == dict
    assert type(edge_info) == dict
    
    t, node_info, edge_info = git2net.get_coediting_network(sqlite_db_file)
    assert type(t) == pp.TemporalNetwork
    assert type(node_info) == dict
    assert type(edge_info) == dict
    
    n, node_info, edge_info = git2net.get_coauthorship_network(sqlite_db_file)
    assert type(n) == pp.Network
    assert type(node_info) == dict
    assert type(edge_info) == dict
    
    t, node_info, edge_info = git2net.get_bipartite_network(sqlite_db_file)
    assert type(t) == pp.TemporalNetwork
    assert type(node_info) == dict
    assert type(edge_info) == dict
    
    
def test_parse_blame_C():
    with pytest.raises(Exception) as e:
        git2net.extraction._parse_blame_C('invalid') == []
    assert e.value.args[0].startswith("Invalid 'blame_C' supplied.")
    
    assert git2net.extraction._parse_blame_C('') == []
    
    assert git2net.extraction._parse_blame_C('-CCC42') == ['-C', '-C', '-C42']
    
    
def test_mining_options(github_url_short, github_repo_dir, sqlite_db_file):
    git2net.mine_github(github_url_short, github_repo_dir, sqlite_db_file, blame_w=True)
    assert git2net.check_mining_complete(github_repo_dir, sqlite_db_file)
    # add information from other branches
    git2net.mine_github(github_url_short, github_repo_dir, sqlite_db_file, blame_w=True, all_branches=True)
    assert git2net.check_mining_complete(github_repo_dir, sqlite_db_file)
    
    
def test_check_mailmap(git_repo_dir):
    git_repo = pydriller.Git(git_repo_dir)

    example = ('Joe', 'joe@example.com')
    result = ('Joe R. Developer', 'joe@example.com')
    assert git2net.extraction._check_mailmap(*example, git_repo) == result

    example = ('Test', 'jane@laptop.(none)')
    result = ('Jane Doe', 'jane@example.com')
    assert git2net.extraction._check_mailmap(*example, git_repo) == result

    example = ('Jane Doe', 'jane@desktop.(none)')
    result = ('Jane Doe', 'jane@example.com')
    assert git2net.extraction._check_mailmap(*example, git_repo) == result

    example = ('Joe', 'bugs@example.com')
    result = ('Joe R. Developer', 'joe@example.com')
    assert git2net.extraction._check_mailmap(*example, git_repo) == result

    example = ('Jane', 'bugs@example.com')
    result = ('Jane Doe', 'jane@example.com')
    assert git2net.extraction._check_mailmap(*example, git_repo) == result

    example = ('SomeoneElse', 'bugs@example.com')
    result = example
    assert git2net.extraction._check_mailmap(*example, git_repo) == result

    example = ('invalid', 'invalid@invalid.com')
    result = example
    assert git2net.extraction._check_mailmap(*example, git_repo) == result

    example = ('--', '--')
    result = example
    assert git2net.extraction._check_mailmap(*example, git_repo) == result

    example = ('-', '-')
    result = example
    assert git2net.extraction._check_mailmap(*example, git_repo) == result