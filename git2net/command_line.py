from .extraction import mine_git_repo
from .visualisation import get_line_editing_paths
from .visualisation import get_commit_editing_paths
from .visualisation import get_coediting_network
from .visualisation import get_coauthorship_network
from .visualisation import get_bipartite_network

import argparse
import os
import datetime
    
def main():
    parser = argparse.ArgumentParser(description='Allows git2net to be used from the command line.')

    subparsers = parser.add_subparsers(dest='command',
        help='Mine a repository or create graph projections from a database.')
    mine = subparsers.add_parser('mine', description="""Mine a given git repository. Information
        on commits and edits will be written to an SQLite database at the provided path.""")
    graph = subparsers.add_parser('graph', description="""Generate graph projections from commit and
        edit information stored in a provided SQLite database. The database needs to be created
        using the 'mine' command in git2net. Graphs will be output as csv files at the given path.
    """)

    mine.add_argument('repo', help='Path to a local copy of the git reposity that will be mined.',
                      type=str)
    mine.add_argument('database', help='Path to the database where results will be stored.',
                      type=str)


    # "mine" options
    mine.add_argument('--commits',
        help='Path to text file with list of commits to mine. Mines all commits if not provided (default).',
        dest='commits', type=str, default=None)
    mine.add_argument('--use-blocks',
        help='Compare added and deleted blocks of code rather than lines.', dest='use_blocks',
        action='store_true', default=False)
    mine.add_argument('--numprocesses',
        help='Number of CPU cores used for multi-core processing. Defaults to number of CPU cores.',
        default=os.cpu_count(), type=int, dest='numprocesses')
    mine.add_argument('--chunksize', help='Chunk size to be used in multiprocessing mapping.',
        default=1, type=int, dest='chunksize')
    mine.add_argument('--exclude', help='Exclude path prefixes in given file.', type=str,
        default=None, dest='exclude')
    mine.add_argument('--blame-C', help="Git blame -C option. To not use -C provide '' (default)", type=str,
        dest='blame_C', default='')
    mine.add_argument('--blame-w', help='ignore whitespaces in git blame (-w option)', dest='blame_w',
        action='store_true', default=False)
    mine.add_argument('--max-modifications', help='Do not process commits with more than given ' +
        'number of modifications. Use 0 to disable.', dest='max_modifications', default=0, type=int)
    mine.add_argument('--timeout', help='Stop processing commit after timeout. Use 0 to disable.',
        default=0, type=int, dest='timeout')
    mine.add_argument('--extract-text',
        help='Extract the commit message and line texts.', dest='extract_text',
        action='store_true', default=False)
    mine.add_argument('--extract_complexity',
        help='Extract cyclomatic complexity and length of file (computationally expensive).', dest='extract_complexity',
        action='store_true', default=False)
    mine.add_argument('--extract_merges',
        help='Process merges.', dest='extract_merges',
        action='store_true', default=True)
    mine.add_argument('--extract_merge_deletions',
        help='Extract lines that are not accepted during a merge as deletions.', dest='extract_merge_deletions',
        action='store_true', default=False)
    
    # "graph" options
    subparsers_graph = graph.add_subparsers(dest='projection',
                                            help='Type of graph projection.')

    graph_coedit = subparsers_graph.add_parser('coedit',
                                               description='Co-editing network projection.')
    graph_bipartite = subparsers_graph.add_parser('bipartite',
                                                  description='Bipartite network projection.')
    graph_coauthor = subparsers_graph.add_parser('coauthor',
                                                 description='Co-authorship network projection.')
    graph_commit_editing = subparsers_graph.add_parser('commit_editing',
                                            description='Commit editing DAG projection.')
    graph_line_editing = subparsers_graph.add_parser('line_editing',
                                            description='Line editing DAG projection.')

    graph_priojections = [graph_coedit, graph_bipartite, graph_coauthor,
                          graph_commit_editing, graph_line_editing]
    has_time = [graph_coedit, graph_bipartite, graph_coauthor, graph_commit_editing]
    has_filename = [graph_commit_editing, graph_line_editing]

    for sp in graph_priojections:
        sp.add_argument('database', help='Path to the database previously mined with git2net.',
                        type=str)
        sp.add_argument('csvfile', help='Path where the resulting graph will be stored as csv.',
                        type=str)

    for sp in has_time:
        sp.add_argument('--time_from', help='Start time in format "%%Y-%%m-%%d %%H:%%M:%%S".',
                        dest='time_from', default=None, type=str)
        sp.add_argument('--time_to', help='Start time in format "%%Y-%%m-%%d %%H:%%M:%%S".',
                        dest='time_to', default=None, type=str)

    for sp in has_filename:
        sp.add_argument('--filename', help="""Path to file in the repository for which
                                              commit-editing paths are extracted.""",
                        dest='filename', default=None, type=str)

    args = parser.parse_args()
    
    if not args.command:
        raise Exception('Requires command argument: "mine" or "graph".')
    
    if args.commits:
        with open(args.commits, 'r') as f:
            args.commits = f.read().split('\n')
            args.commits = [x for x in args.commits if len(x) > 0]
    else:
        args.commits = []
    if args.exclude:
        with open(args.exclude, 'r') as f:
            args.exclude = f.read().split('\n')
            args.exclude = [x for x in args.exclude if len(x) > 0]
    else:
        args.exclude = []
    
    if args.command == 'graph':
        if args.projection in has_time:
            if args.time_from:
                args.time_from = datetime.strptime(args.time_from, '%Y-%m-%d %H:%M:%S')
            if args.time_to:
                args.time_to = datetime.strptime(args.time_to, '%Y-%m-%d %H:%M:%S')

    if args.command == 'mine':
        mine_git_repo(args.repo, args.database, commits=args.commits, use_blocks=args.use_blocks,
                      no_of_processes=args.numprocesses, chunksize=args.chunksize,
                      exclude=args.exclude, blame_C=args.blame_C, blame_w=args.blame_w,
                      max_modifications=args.max_modifications, timeout=args.timeout,
                      extract_text=args.extract_text, extract_complexity=args.extract_complexity,
                      extract_merges=args.extract_merges,
                      extract_merge_deletions=args.extract_merge_deletions)
    elif args.command == 'graph':
        if args.projection == 'commit_editing':
            _, d, _, _ = get_commit_editing_paths(args.database, filename=args.filename,
                                                  time_from=args.time_from, time_to=args.time_to)
            d.write_file(args.csvfile)
        elif args.projection == 'line_editing':
            if args.filename:
                args.filename = [args.filename]
            _, d, _, _ = get_line_editing_paths(args.database, file_paths=args.filename)
            print(d)
            d.write_file(args.csvfile)
        elif args.projection == 'coedit':
            t, _, _ = get_coediting_network(args.database,
                                            time_from=args.time_from, time_to=args.time_to)
            t.write_file(args.csvfile)
        elif args.projection == 'coauthor':
            n, _, _ = get_coauthorship_network(args.database,
                                               time_from=args.time_from, time_to=args.time_to)
            n.write_file(args.csvfile)
        elif args.projection == 'bipartite':
            t, _, _ = get_bipartite_network(args.database,
                                            time_from=args.time_from, time_to=args.time_to)
            t.write_file(args.csvfile)