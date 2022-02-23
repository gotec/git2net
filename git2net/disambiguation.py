import gambit
import sqlite3
import pandas as pd


def disambiguate_aliases_db(sqlite_db_file, method='gambit', **quargs):
    """
    Disambiguates author aliases in a given SQLite database mined with `git2net`.
    The disambiguation is performed using the Python package `gambit`.
    Internally, `disambiguate_aliases_db` calls the function `gambit.disambiguate_aliases <https://github.com/gotec/gambit/blob/main/gambit/main.py>`_.
    
    :param str sqlite_db_file: path to SQLite database
    :param str method: disambiguation method from {"gambit", "bird", "simple"}
    :param \**quargs: hyperparameters for the gambit and bird algorithms;
        **gambit**:
        thresh (*float*) – similarity threshold  from interval 0 to 1,
        sim (*str*) – similarity measure from {'lev', 'jw'},
        **bird**:
        thresh (*float*) – similarity threshold  from interval 0 to 1
                          
    :return:
        creates new column with unique `author_id` in the `commits` table of the provided database
    """

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
