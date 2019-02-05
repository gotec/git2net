#%%
import importlib
import git2net
importlib.reload(git2net)
import pydriller

post_line_num = 102
file_name = 'test.py'
commit_hash = 'db0e20b413363cb93447ed4567bb8e959fc7f306'


git_repo = pydriller.GitRepository('.')
commit = git_repo.get_commit(commit_hash)
mod = commit.modifications[0]


blame = git_repo.git.blame(commit_hash + '^', '--', 'test.py').split('\n')

blame_fields = blame[post_line_num - 1].split(' ')
original_commit_hash = blame_fields[0].replace('^', '')
original_commit_hash

git2net.get_original_line_number(git_repo, file_name, original_commit_hash, commit_hash + '^', post_line_num)

#%%
pre_(50, 60)

if

#%%
def extrapolate_line_mapping(mapping, line_no):
    # 'removekey' function source: https://stackoverflow.com/questions/5844672
    def removekey(d, key):
        r = dict(d)
        del r[key]
        return r

    mapping = removekey(mapping, False)

    if line_no in mapping.keys():
        return mapping[line_no]
    elif line_no < min(mapping.keys()):
        return line_no
    elif line_no > max(mapping.keys()):
        return line_no + mapping[max(mapping.keys())] - max(mapping.keys())
    else:
        raise Exception("Unexpected error in 'get_original_line_number'.")


post_to_pre = {10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 15: 12, False: 187, 16: 14, 17: 15, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 26: 25, 27: 26, 28: 27, 29: 28, 30: 29, 31: 30, 32: 31, 33: 32, 34: 33, 35: 34, 36: 35, 37: 36, 38: 37, 39: 38, 40: 39, 41: 40, 42: 41, 43: 42, 44: 43, 45: 44, 46: 45, 53: 46, 54: 47, 55: 48, 56: 49, 57: 50, 58: 51, 59: 52, 60: 53, 61: 54, 62: 55, 63: 56, 64: 57, 65: 58, 66: 59, 67: 60, 68: 61, 69: 62, 70: 63, 71: 64, 72: 65, 73: 66, 74: 67, 75: 68, 76: 69, 77: 70, 78: 71, 79: 72, 80: 73, 81: 74, 82: 75, 83: 76, 84: 77, 85: 78, 86: 79, 87: 80, 88: 81, 89: 82, 90: 83, 91: 84, 92: 85, 93: 86, 94: 87, 95: 88, 96: 89, 97: 90, 98: 91, 99: 92, 102: 93, 103: 94, 104: 95, 106: 96, 107: 97, 108: 98, 109: 99, 110: 100, 111: 101, 112: 102, 113: 103, 114: 104, 115: 105, 124: 106, 125: 107, 126: 108, 127: 109, 128: 110, 129: 111, 130: 112, 131: 113, 132: 114, 133: 115, 138: 116, 139: 117, 140: 118, 146: 119, 147: 120, 148: 121, 149: 122, 150: 123, 151: 124, 152: 125, 153: 126, 154: 127, 155: 128, 156: 129, 158: 130, 159: 131, 160: 132, 161: 133, 162: 136, 163: 137, 164: 138, 165: 139, 166: 140, 167: 141, 168: 142, 169: 143, 170: 144, 171: 145, 172: 147, 173: 148, 174: 149, 175: 150, 176: 151, 177: 152, 178: 172, 179: 176, 180: 179, 181: 180, 182: 181, 187: 182, 188: 183, 189: 184, 191: 185}

extrapolate_line_mapping(post_to_pre, 192)

#%%
print(max(post_to_pre.keys()), ': ', post_to_pre[max(post_to_pre.keys())])

#%%
'6a844a712042c3ce688a2060d8ae691b9ab86a32'
364
'15e0d8e497ee8d91e67b081ef9444bb7d3a6e9d9'
361