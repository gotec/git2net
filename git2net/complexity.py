import multiprocessing
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import time
from tqdm import tqdm
import pydriller
from pygments import lexers
import lizard


git_init_lock = multiprocessing.Lock()

def _compute_halstead_effort(filename, source_code):
    """Computes the Healstead effort (https://en.wikipedia.org/wiki/Halstead_complexity_measures)
       for a given pydriller modification object based on an algorithm adapted from
       https://github.com/priv-kweihmann/multimetric."""

    _needles_operators = [
        "Token.Name.Class",
        "Token.Name.Decorator",
        "Token.Name.Entity",
        "Token.Name.Exception",
        "Token.Name.Function.Magic",
        "Token.Name.Function",
        "Token.Name.Label",
        "Token.Name.Tag",
        "Token.Operator.Word",
        "Token.Operator",
        "Token.Punctuation",
        "Token.String.Affix",
        "Token.String.Delimiter"]

    _needles_operands = [
        "Token.Literal.Date",
        "Token.Literal.String.Double",
        "Token.Literal.String",
        "Token.Literal.Number.Bin",
        "Token.Literal.Number.Float",
        "Token.Literal.Number.Hex",
        "Token.Literal.Number.Integer.Long",
        "Token.Literal.Number.Integer",
        "Token.Literal.Number.Oct",
        "Token.Literal.Number",
        "Token.Name",
        "Token.Name.Attribute",
        "Token.Name.Builtin.Pseudo",
        "Token.Name.Builtin",
        "Token.Name.Constant",
        "Token.Name.Variable.Class",
        "Token.Name.Variable.Global",
        "Token.Name.Variable.Instance",
        "Token.Name.Variable.Magic",
        "Token.Name.Variable",
        "Token.Name.Other",
        "Token.Number.Bin",
        "Token.Number.Float",
        "Token.Number.Hex",
        "Token.Number.Integer.Long",
        "Token.Number.Integer",
        "Token.Number.Oct",
        "Token.Number",
        "Token.String.Char",
        "Token.String.Double",
        "Token.String.Escape",
        "Token.String.Heredoc",
        "Token.String.Interpol",
        "Token.String.Other",
        "Token.String.Regex",
        "Token.String.Single",
        "Token.String.Symbol"]
    
    # Initialise a lexer for the programming language identified from m.filename.
    try:
        _lexer = lexers.get_lexer_for_filename(filename)
    except lexers.ClassNotFound:
        return 0

    # Parse the source code before the change.
    tokens = list(_lexer.get_tokens(source_code))
    
    # Count the number of operators
    operators_counter =  [str(x[1]) for x in tokens if str(x[0]) in _needles_operators]
    eta_1 = max(1, len(set(operators_counter)))
    N_1 = max(1, len(operators_counter))
    
    # Count the number of operands
    operands_counter = [str(x[1]) for x in tokens if str(x[0]) in _needles_operands]
    eta_2 = max(1, len(set(operands_counter)))
    N_2 = max(1, len(operands_counter))
    
    # Compute the Halstead effort based on these values
    HE = (eta_1/2 * N_2/eta_2) * ((N_1+N_2) * np.log2(eta_1+eta_2))
    
    return HE


def _compute_complexity_measures(args):
    """
    Computes a set of complexity measures for a given commit/file combination.
    
    :param dict args: dictionary with the following key/value pairs:
        * **git_repo_dir** (*str*) – path to the git repository that is analysed
        * **commit_hash** (*str*) – hash of the commit that is processed
        * **old_path** (*str*) – path to the analysed file before the commit
        * **new_path** (*str*) – path to the analysed file after the commit
        * **events** (*int*) – number of edit events in the commit/file pair
        * **levenshtein_distance** (*int*) – total Levenshtein distance in the commit/file pair
    
    :return:
        *pandas.DataFrame* – dataframe containing identifying information and the computed
            complexity for the commit/file combination.
    """
    
    filename_old = args['old_path'].split('/')[-1]
    filename_new = args['new_path'].split('/')[-1]
    if filename_new != 'None':
        filename = filename_new
    else:
        filename = filename_old
    
    result = {'commit_hash': args['commit_hash'],
              'old_path': args['old_path'],
              'new_path': args['new_path'],
              'events': args['events'],
              'levenshtein_distance': args['levenshtein_distance'],
              'HE_pre': None,
              'CCN_pre': None,
              'NLOC_pre': None,
              'TOK_pre': None,
              'FUN_pre': None,
              'HE_post': None,
              'CCN_post': None,
              'NLOC_post': None,
              'TOK_post': None,
              'FUN_post': None,
              'HE_delta': None,
              'CCN_delta': None,
              'NLOC_delta': None,
              'TOK_delta': None,
              'FUN_delta': None}
    
    with git_init_lock:
        pydriller_repo = pydriller.Git(args['git_repo_dir'])
        pydriller_commit = pydriller_repo.get_commit(args['commit_hash'])
        
    found = False
    for m in pydriller_commit.modified_files:
        if m.filename == filename:
            found = True
            break
    
    if found:
        if pd.notnull(m.source_code_before):
            result['HE_pre'] = _compute_halstead_effort(m.old_path, m.source_code_before)
            l_before = lizard.analyze_file.analyze_source_code(m.old_path, m.source_code_before)
            result['CCN_pre'] = l_before.CCN
            result['NLOC_pre'] = l_before.nloc
            result['TOK_pre'] = l_before.token_count
            result['FUN_pre'] = len(l_before.function_list)
        else: 
            result['HE_pre'] = 0
            result['CCN_pre'] = 0
            result['NLOC_pre'] = 0
            result['TOK_pre'] = 0
            result['FUN_pre'] = 0
                
        if pd.notnull(m.source_code):
            result['HE_post'] = _compute_halstead_effort(m.new_path, m.source_code)
            l_after = lizard.analyze_file.analyze_source_code(m.new_path, m.source_code)
            result['CCN_post'] = l_after.CCN
            result['NLOC_post'] = l_after.nloc
            result['TOK_post'] = l_after.token_count
            result['FUN_post'] = len(l_after.function_list)
        else: 
            result['HE_post'] = 0
            result['CCN_post'] = 0
            result['NLOC_post'] = 0
            result['TOK_post'] = 0
            result['FUN_post'] = 0
    
    result['HE_delta'] = result['HE_post'] - result['HE_pre']
    result['CCN_delta'] = result['CCN_post'] - result['CCN_pre']
    result['NLOC_delta'] = result['NLOC_post'] - result['NLOC_pre']
    result['TOK_delta'] = result['TOK_post'] - result['TOK_pre']
    result['FUN_delta'] = result['FUN_post'] - result['FUN_pre']
    
    result_df = pd.DataFrame(result, index=[0])
        
    return result_df


def compute_complexity(git_repo_dir, sqlite_db_file, no_of_processes=os.cpu_count(), read_chunksize = 1e6,
                         write_chunksize = 100):
    """
    Computes complexity measures for all mined commit/file combinations in a given database. Computing
    complexities for merge commits is currently not supported.
    
    :param str git_repo_dir: path to the git repository that is analysed
    :param str sqlite_db_file: path to the SQLite database containing the mined commits
    :param str no_of_processes: number of parallel processes that are spawned
    :param str read_chunksize: number of commit/file combinations that are processed at once
    :param str write_chunksize: number of commit/file combinations for which complexities are written at once
    
    :return:
        adds table `complexity` containing computed complexity measures for all commit/file combinations.
    """
    
    
    # Identify merge commits. These are not supported for the complexity computation.
    with sqlite3.connect(sqlite_db_file) as con:
        merge_commits = set(pd.read_sql("""SELECT hash
                                           FROM commits
                                           WHERE merge==1""", con).hash)
    
    
    with sqlite3.connect(sqlite_db_file) as con:
        cur = con.cursor()
        tables = [i[0] for i in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]

        # Identify which commit/file combinations have already been processed. These can be skipped.
        if 'complexity' in tables:
            complexity = pd.read_sql("""SELECT commit_hash, old_path, new_path FROM complexity""",
                                            con)
            already_done = set((row.commit_hash, row.old_path, row.new_path) \
                               for idx, row in complexity.iterrows())

        else:
            already_done = set()

        # Go through edits in chunks.
        edits_chunks = []
        for edits_chunk in pd.read_sql("""SELECT commit_hash,
                                                 old_path,
                                                 new_path,
                                                 original_commit_deletion,
                                                 levenshtein_dist as levenshtein_distance
                                          FROM edits""", con, chunksize = int(read_chunksize)):
            # Ensure the types are correct.
            edits_chunk['commit_hash'] = edits_chunk['commit_hash'].astype(str)
            edits_chunk['old_path'] = edits_chunk['old_path'].astype(str)
            edits_chunk['new_path'] = edits_chunk['new_path'].astype(str)
            edits_chunk['original_commit_deletion'] = edits_chunk['original_commit_deletion'].astype(str)
            edits_chunk['levenshtein_distance'] = edits_chunk['levenshtein_distance'].astype(float)
            
            # Exclude all edits from merge commits.
            edits_chunk = edits_chunk.loc[~edits_chunk.commit_hash.isin(merge_commits)]
            
            if len(edits_chunk) == 0:
                continue

            # Complexity can only be computed on a file level. Therefore, we aggregate all edits
            # from the same file. The number of edit events (addition, deletion, replacement)
            # yeilds the event count. The sum of Levenshtein distances over the edited lines in
            # the file yields the total Levenshtein distance for the file.
            edits_chunk = edits_chunk.groupby(['commit_hash', 'old_path', 'new_path']) \
                                     .agg({'levenshtein_distance': ['count', 'sum']})

            # The results from the individual edit chunks are written into a list. Writing them
            # directly to the database is not possible as by processing them in chunks, the edits
            # from the same commit/file combination might appear in different chunks. However,
            # by aggregating the individual lines for all commit/file combinations we substantially
            # reduce the amount of memory required. Hence, given this works with reasonable memory
            # requirements.
            edits_chunks.append(edits_chunk)

    # We concenate the results from all processed chunks into a single dataframe.
    total_compute = pd.concat(edits_chunks).reset_index()
    total_compute.columns = ['commit_hash', 'old_path', 'new_path', 'events', 'levenshtein_distance']

    # We aggregate this dataframe once again to ensure edits that were processed in different chunks
    # are appropriately combined.
    total_compute = total_compute.groupby(['commit_hash', 'old_path', 'new_path']) \
                                 .agg({'events': 'sum',
                                       'levenshtein_distance': 'sum'}).reset_index()
    
    # So far, we have only computed the event count and the Levenshtein distance. We now compute the
    # remaining complexity measures for all commit/file combinations. For efficiency, we parallelise
    # this operation, and only process those commit/file combinations not already present in the
    # database.
    args_pool = [{'git_repo_dir': git_repo_dir, 'commit_hash': row.commit_hash, 'old_path': row.old_path,
                  'new_path': row.new_path, 'events': row.events, 'levenshtein_distance': row.levenshtein_distance}
                     for idx, row in total_compute.iterrows()
                     if not (row.commit_hash, row.old_path, row.new_path) in already_done]
    
    def _init(git_repo_dir, git_init_lock_):
        global git_init_lock
        git_init_lock = git_init_lock_
        
    with multiprocessing.Pool(no_of_processes, initializer=_init,
                              initargs=(git_repo_dir,git_init_lock)) as p:
        results = []
        with tqdm(total=len(args_pool), desc='complexity computation') as pbar:
            for result in p.imap_unordered(_compute_complexity_measures, args_pool, chunksize=1):
                results.append(result)

                # We write the results to the database. If results are already present, we append
                # the new results to them.
                if len(results) >= write_chunksize:
                    results = pd.concat(results, axis=0, copy=False)
                    
                    results = results[['commit_hash', 'old_path', 'new_path',
                                       'events', 'levenshtein_distance',
                                       'HE_pre', 'HE_post', 'HE_delta',
                                       'CCN_pre', 'CCN_post', 'CCN_delta',
                                       'NLOC_pre', 'NLOC_post', 'NLOC_delta',
                                       'TOK_pre', 'TOK_post', 'TOK_delta',
                                       'FUN_pre', 'FUN_post', 'FUN_delta']]
                    
                    with sqlite3.connect(sqlite_db_file) as con:
                        results.to_sql('complexity', con, if_exists='append', index=False)
                    results=[]

                pbar.update(1)

    # As we write the results in chunks, some might remain unwritten in the loop above. We write
    # them now.
    if len(results) != 0:
        results = pd.concat(results, axis=0, copy=False)
        with sqlite3.connect(sqlite_db_file) as con:
            results.to_sql('complexity', con, if_exists='append', index=False)
