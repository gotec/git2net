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

def chunk_size(lines, k):
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


def pre_to_post(git_repo, commit, mod):    
    
    print('=======================')
    print('File: ' + mod.filename)
    print('=======================')

    diff = git_repo.parse_diff(mod.diff)
    print('diff = ' + mod.diff)
    
    deleted_lines = { x[0]:x[1] for x in git_repo.parse_diff(mod.diff)['deleted'] }
    added_lines = { x[0]:x[1] for x in git_repo.parse_diff(mod.diff)['added'] } 
    max_added = max(added_lines.keys())
    max_deleted = max(deleted_lines.keys())

    # create mapping between pre and post edit line numbers
    left_to_right = {}
    right = 1
    left = 1

    no_right_inc = 0
    both_inc = 0
    jump_right = 0

    while left <= max(max_added, max_deleted):

        # compute size of added and deleted chunks
        chunk_added = chunk_size(added_lines, right)
        chunk_deleted = chunk_size(deleted_lines, left)

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

    return left_to_right


def blame_modification(git_repo, commit, mod):
    path = mod.new_path
    deleted_lines = { x[0]:x[1] for x in git_repo.parse_diff(mod.diff)['deleted'] }
    added_lines = { x[0]:x[1] for x in git_repo.parse_diff(mod.diff)['added'] }
    try:
        blame = git_repo.git.blame(commit.hash + '^', '--', path).split('\n')
        for num_line, line in deleted_lines.items():
            blame_fields = blame[num_line - 1].split(' ')
            buggy_commit_hash = blame_fields[0].replace('^', '')
            buggy_commit = git_repo.get_commit(buggy_commit_hash)
            print('Previous commit: \t' + buggy_commit_hash)
            print('Author: \t\t' + buggy_commit.author.email)
            print('Date: \t\t\t' + buggy_commit.author_date.strftime('%Y-%m-%d %H:%M:%S'))
            print('blame\t\t\t= ', blame[num_line-1])

            left_to_right = pre_to_post(git_repo, commit, mod)
            right_to_left = { v:k for k,v in left_to_right.items() if v != False }

            print('deleted line num\t= ', num_line)
            print('deleted line\t\t= ', line)
            
            print('added line num\t\t= ', left_to_right[num_line])
            if left_to_right[num_line] in added_lines:
                print('added line\t\t= ', added_lines[left_to_right[num_line]])
            else:
                print('removed line')
            print('-----------------------------')

    except GitCommandError:
        logger.debug("Could not found file %s in commit %s. Probably a double rename!", mod.filename,
                        commit.hash)



#%%
c = git_repo.get_commit('322655441586e728cd68255ed40223485334cacd')
for c_m in c.modifications:
    blame_modification(git_repo, c, c_m)

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
