#%%
import git2net
import pathpy as pp
import pytest
import pydriller
import numpy as np
import lizard
import os

#%%
@pytest.yield_fixture(scope="module")
def repo_string():
    yield 'test_repos/test_repo_1'

def test_get_commit_dag(repo_string):
    dag = git2net.get_commit_dag(repo_string)
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
                      ('94a9da2', '83214bc')]
    assert list(dag.edges.keys()) == expected_edges

def test_extract_edits_1(repo_string):
    commit_hash = 'b17c2c321ce8d299de3d063ca0a1b0b363477505'
    filename = 'first_lines.txt'

    git_repo = pydriller.GitRepository(repo_string)
    commit = git_repo.get_commit(commit_hash)
    for mod in commit.modifications:
        if mod.filename == filename:
            df = git2net.extract_edits(git_repo, commit, mod, use_blocks=False, blame_C='CCC4')
    assert len(df) == 3
    assert df.at[0, 'original_commit_addition'] == 'e4448e87541d19d139b9d033b2578941a53d1f97'
    assert df.at[1, 'original_commit_addition'] == '6b531fcb57d5b9d98dd983cb65357d82ccca647b'
    # note that this line is not found as it does not end with a newline while the original line did
    assert df.at[2, 'original_commit_addition'] is None

def test_extract_edits_2(repo_string):
    commit_hash = 'b17c2c321ce8d299de3d063ca0a1b0b363477505'
    filename = 'first_lines.txt'

    git_repo = pydriller.GitRepository(repo_string)
    commit = git_repo.get_commit(commit_hash)
    for mod in commit.modifications:
        if mod.filename == filename:
            df = git2net.extract_edits(git_repo, commit, mod, use_blocks=True, blame_C='CCC4')
    assert len(df) == 1
    assert df.at[0, 'original_commit_addition'] == 'not available with use_blocks'

def test_identify_edits(repo_string):
    commit_hash = 'f343ed53ee64717f85135c4b8d3f6bd018be80ad'
    filename = 'text_file.txt'

    git_repo = pydriller.GitRepository(repo_string)
    commit = git_repo.get_commit(commit_hash)
    for x in commit.modifications:
        if x.filename == filename:
            mod = x

    parsed_lines = git_repo.parse_diff(mod.diff)

    deleted_lines = { x[0]:x[1] for x in parsed_lines['deleted'] }
    added_lines = { x[0]:x[1] for x in parsed_lines['added'] }

    _, edits = git2net.identify_edits(deleted_lines, added_lines, use_blocks=False)
    assert list(edits.type) == ['deletion', 'replacement', 'deletion', 'replacement', 'addition',
                                'addition', 'addition']


def test_process_commit(repo_string):
    commit_hash = 'f343ed53ee64717f85135c4b8d3f6bd018be80ad'
    args = {'repo_string': repo_string, 'commit_hash': commit_hash, 'use_blocks': False,
             'exclude_paths': [], 'blame_C': '-C'}
    res_dict = git2net.process_commit(args)
    assert list(res_dict.keys()) == ['commit', 'edits']

####################################################################################################

# repo_string = 'test_repos/test_repo_1'
# git_repo = pydriller.GitRepository(repo_string)
# commit = git_repo.get_commit('deff6c8997991a0b559cee3ef70af223fbb85ec8')
# edited_file_path = 'text_file.txt'

# edited_file_paths = git2net.get_edited_file_paths_since_split(git_repo, commit)
# for edited_file_path in edited_file_paths:
#     modification_info = {}
#     file_contents = git_repo.git.show('{}:{}'.format(commit.hash, edited_file_path))
#     l = lizard.analyze_file.analyze_source_code(edited_file_path, file_contents)

#     modification_info['filename'] = edited_file_path.split(os.sep)[-1]
#     modification_info['new_path'] = edited_file_path
#     modification_info['old_path'] = edited_file_path
#     modification_info['cyclomatic_complexity_of_file'] = l.CCN
#     modification_info['lines_of_code_in_file'] = l.nloc
#     modification_info['modification_type'] = 'merge_self_accept'

#     edits_info = git2net.extract_edits_merge(git_repo, commit, modification_info, use_blocks=False, blame_C='-C')

####################################################################################################

# repo_string = 'test_repos/test_repo_1'
# git_repo = pydriller.GitRepository(repo_string)
# commit = git_repo.get_commit('96025072a3e1b2f466ef56053bbdf4c9c0e927f0')
# edited_file_path = 'text_file.txt'

# for modification in commit.modifications:
#     modification_info = {}
#     modification_info['filename'] = modification.filename
#     modification_info['new_path'] = modification.new_path
#     modification_info['old_path'] = modification.old_path
#     modification_info['cyclomatic_complexity_of_file'] = modification.complexity
#     modification_info['lines_of_code_in_file'] = modification.nloc
#     modification_info['modification_type'] = modification.change_type.name

#     edits_info = git2net.extract_edits_merge(git_repo, commit, modification_info, use_blocks=True, blame_C='-C')

####################################################################################################


#%%
import git2net
import pathpy as pp
import pytest
import pydriller
import numpy as np
import lizard
import os

def test_extract_editing_paths(repo_string):
    git2net.mine_git_repo(repo_string, 'test.db', blame_C='-C4')
    commit_hashes = [x.hash for x in pydriller.GitRepository(repo_string).get_list_commits()]#[29]#18]
    dag, paths, node_info, edge_info = git2net.extract_editing_paths('test.db', commit_hashes, with_start=True)
    return dag, paths, node_info, edge_info

dag, paths, node_info, edge_info = test_extract_editing_paths('test_repos/test_repo_1')
pp.visualisation.plot(dag, width=1000, height=1000, node_color=node_info['colors'],
                      edge_color=edge_info['colors'], edge_width=1.0, label_opacity=0, node_size=8.0)
for edge in edge_info['colors']:
    print(edge_info['colors'][edge])
    if edge_info['colors'][edge] == 'white':
        edge_info['colors'][edge] = 'black'
pp.visualisation.export_html(dag, 'editing_paths.html', width=1500, height=1500, node_color=node_info['colors'],
                      edge_color=edge_info['colors'], edge_width=1.0, label_opacity=0, node_size=8.0)

























# def test_identify_file_renaming(repo_string):
#     dag, aliases = git2net.identify_file_renaming(repo_string)
#     return dag

# dag = test_identify_file_renaming('test_repos/test_repo_1')
# pp.visualisation.plot(dag, width=1500, height=1500)

#%%


#%%
