import gambit
import sqlite3
import pandas as pd

def disambiguate_aliases_db(sqlite_db_file, method='gambit', **quargs):
    
    # thresh=.95, sim='lev'
    
    with sqlite3.connect(sqlite_db_file) as con:
        aliases = pd.read_sql("""SELECT author_name AS alias_name,
                                        author_email AS alias_email
                                 FROM commits""", con).drop_duplicates()
        
    aliases = gambit.disambiguate_aliases(aliases, method=method, **quargs)
    
    with sqlite3.connect(sqlite_db_file) as con:
        cur = con.cursor()

        cols = [i[1] for i in cur.execute('PRAGMA table_info(commits)')]

        if 'author_id' not in cols:
            cur.execute("""ALTER TABLE commits
                           ADD COLUMN author_id""")
        con.commit()
        for idx, row in aliases.iterrows():
            cur.execute("""UPDATE commits
                           SET author_id = :author_id
                           WHERE author_name IS :author_name
                           AND author_email IS :author_email""",
                        {'author_id': row.author_id,
                         'author_name': row.alias_name,
                         'author_email': row.alias_email})
            con.commit()
