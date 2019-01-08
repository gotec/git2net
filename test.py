#%%
import pathpy as pp
import pydriller as pydriller
import sqlite3
import pandas as pd
from pydriller.git_repository import GitCommandError

repo_string = '~/pathpy/'
git_repo = pydriller.GitRepository(repo_string)

df_commits = pd.DataFrame()
df_modifications = pd.DataFrame()

#%%
c = git_repo.get_commit('322655441586e728cd68255ed40223485334cacd')
for c_m in c.modifications:
    blame_modification(git_repo, c, c_m)    

#%%
def get_added_line_num(added_lines, deleted_lines, pre_line_num):
    post_line_num = pre_line_num
    """print('Added Lines')
    print('--------------')
    for n, l in added_lines.items():
        if n < pre_line_num:
            print(str(n) + ': ' + l)

    print('Deleted Lines')
    print('--------------')
    for n, l in deleted_lines.items():
        if n < pre_line_num:
            print(str(n) + ': ' + l)         """
    
    print('Deleted Lines')
    print('--------------')
    for x in deleted_lines:
        if x < pre_line_num:
            print(str(x) + ': ' + deleted_lines[x])
            post_line_num -= 1

    print('Added Lines')
    print('--------------')
    for x in added_lines:
        if x < post_line_num:
            print(str(x) + ': ' + added_lines[x])
            post_line_num += 1

    return post_line_num


def blame_modification(git_repo, commit, mod):
    
    path = mod.new_path
    
    print('=======================')
    print('File: ' + mod.filename)
    print('=======================')

    diff = git_repo.parse_diff(mod.diff)
    deleted_lines = { x[0]:x[1] for x in git_repo.parse_diff(mod.diff)['deleted'] }
    added_lines = { x[0]:x[1] for x in git_repo.parse_diff(mod.diff)['added'] }
    # print(added_lines)

    try:
        blame = git_repo.git.blame(commit.hash + '^', '--', path).split('\n')
        for num_line, line in deleted_lines.items():
            blame_fields = blame[num_line - 1].split(' ')
            print('deleted line num = ', num_line)
            print('deleted line = ', line)
            added_line_num = get_added_line_num(added_lines, deleted_lines, num_line)
            print('added line num = ', added_line_num)
            print('added line = ', added_lines[added_line_num])
            print('blame = ', blame[num_line-1])
            buggy_commit_hash = blame_fields[0].replace('^', '')
            buggy_commit = git_repo.get_commit(buggy_commit_hash)
            print('----------------------------------')
            print('Previous commit: ' + buggy_commit_hash)
            print('Author: ' + buggy_commit.author.email)
            print('Date: ' + buggy_commit.author_date.strftime('%Y-%m-%d %H:%M:%S'))
            for buggy_mod in buggy_commit.modifications:
                if buggy_mod.filename == mod.filename:                          
                    prev_diff = git_repo.parse_diff(buggy_mod.diff)                
                    # for x in prev_diff['added']:
                    #    print(x)

    except GitCommandError:
        logger.debug("Could not found file %s in commit %s. Probably a double rename!", mod.filename,
                        commit.hash)


#%%
def process_commit(commit):
    global df_commits
    global df_modifications
    global git_repo

    # parse commit
    c = {}
    c['hash'] = [commit.hash]
    c['author_email'] = [commit.author.email]
    c['author_name'] = [commit.author.name]
    c['committer_email'] = [commit.committer.email]
    c['committer_name'] = [commit.committer.name]
    c['author_date'] = [commit.author_date.strftime('%Y-%m-%d %H:%M:%S')]
    c['committer_date'] = [commit.committer_date.strftime('%Y-%m-%d %H:%M:%S')]
    # ...
    df_t = pd.DataFrame(c)
    df_commits = pd.concat([df_commits, df_t])    
    
    # parse modification
    for modification in commit.modifications:
        m = {}
        m['hash'] = [commit.hash]
        m['filename'] = [modification.filename]
        m['diff'] = [modification.diff]        
        # ...

        df_t = pd.DataFrame(m)
        df_modifications = pd.concat([df_modifications, df_t])
        print('Current edit')
        print(modification.diff)
        # get previous edits
        for ch in git_repo.get_commits_last_modified_lines(commit):
            print(ch)
            pc = git_repo.get_commit(ch)
            for pcm in pc.modifications:
                if pcm.file == modification.file:
                    print('\t Previous edit')
                    print('\t' + pcm.diff)

        

#%%
for commit in pydriller.RepositoryMining(repo_string).traverse_commits():
    process_commit(commit)

#%%
print(df_commits)

#%%
print(df_modifications)

#%%
print(n)

#%%
pp.visualisation.plot(pp.Network.from_temporal_network(n), width=800, height=800, node_color=colors)
