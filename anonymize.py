#!/usr/bin/python3

import logging
import sqlite3
import os
import argparse
import binascii

import pandas as pd
import progressbar
import numpy as np
import uuid

import hashlib

def hash_path(s, path_map):    
    comps = s.split('/')
    hashed_s = ''
    for x in comps:
        hashed_s += '/' + str(path_map[x.strip()])
    return hashed_s


def anonymize(infile, outfile, map_db):
    con_in = sqlite3.connect(infile)
    con_out = sqlite3.connect(outfile)
    df = pd.read_sql('SELECT * FROM coedits', con_in)

    paths = set(df['mod_filename'])
    for x in df['mod_old_path']:
        for y in x.split('/'):
            paths.add(y.strip())
    for x in df['mod_new_path']:
        for y in x.split('/'):
            paths.add(y.strip())
    path_map = { x:i for i,x in enumerate(paths)}

    df['mod_filename'] = df['mod_filename'].apply(lambda x: path_map[x])
    df['mod_old_path'] = df['mod_old_path'].apply(lambda x: hash_path(x, path_map))
    df['mod_new_path'] = df['mod_new_path'].apply(lambda x: hash_path(x, path_map))
    df.to_sql('coedits', con_out, if_exists='replace')


    id_email_map_con = sqlite3.connect(map_db)

    commit_data = pd.read_sql('SELECT * FROM commits', con_in)
    id_email_map = pd.read_sql('SELECT user_id, email FROM internal_id_email_map', id_email_map_con)

    # Email to lowercase
    commit_data['author_email'] = commit_data.author_email.apply(lambda x: x.lower())
    commit_data['committer_email'] = commit_data.committer_email.apply(lambda x: x.lower())
    id_email_map['email'] = id_email_map.email.apply(lambda x: x.lower())

    # Authors
    commit_data = pd.merge(commit_data, id_email_map, how='left', left_on='author_email', right_on='email')
    cols = commit_data.columns.tolist()
    cols = cols[:2] + cols[-2:] + cols[2:-2]
    commit_data = commit_data[cols]
    commit_data.drop(['email'], axis=1, inplace=True)
    commit_data.rename({'user_id': 'author_id'}, axis=1, inplace=True)

    # Committers
    commit_data = pd.merge(commit_data, id_email_map, how='left', left_on='committer_email', right_on='email')
    cols = commit_data.columns.tolist()
    cols = cols[:7] + cols[-2:] + cols[7:-2]
    commit_data = commit_data[cols]
    commit_data.drop(['email'], axis=1, inplace=True)
    commit_data.rename({'user_id': 'committer_id'}, axis=1, inplace=True)

    # Drop email and name fields
    commit_data.drop(['author_email', 'author_name', 'committer_email', 'committer_name'], axis=1, inplace=True)

    # Output
    commit_data.to_sql('commits', con_out, if_exists='replace')
    

parser = argparse.ArgumentParser(description='Anonymizes SQlite data.')
parser.add_argument('infile', help='path to SQLite DB file to be anonymized.', type=str)
parser.add_argument('outfile', help='path to SQLite DB file that will be written.', type=str)
parser.add_argument('map_file', help='path to SQLite DB file that contains user mapping.', type=str)

args = parser.parse_args()

anonymize(args.infile, args.outfile,  args.map_file)
