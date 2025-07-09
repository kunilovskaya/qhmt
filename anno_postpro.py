"""
02 Jul 2024
Post-process annotated translations.
Put tables with human annotations into assess_out/real/

python3 anno_postpro.py

"""

import sys

import pandas as pd
import argparse
import os
import time
from datetime import datetime
import numpy as np


def generate_proxy(blank, raters=None):
    my_proxy = pd.read_csv(blank, sep='\t')
    my_proxy = my_proxy.drop(['score', 'severity'], axis=1)
    # left skewed (tail on the left), more higher scores
    base_scores = np.random.beta(a=5, b=2, size=len(my_proxy)) * 9 + 1
    ratings = []
    for base in base_scores:
        item_ratings = np.clip(
            np.random.normal(loc=base, scale=1.0, size=len(raters)),  # low noise
            1, 10
        )
        ratings.append(np.round(item_ratings))  # [array([5., 3., 4.])]

    ratings = np.array(ratings)  # convert list of arrays to shape (n_items, n_raters)
    for i, name in enumerate(raters):
        my_proxy[name] = ratings[:, i]

    return my_proxy


def make_dirs(logs=None, make_them=None):
    for i in make_them:
        if i:
            os.makedirs(i, exist_ok=True)
    os.makedirs(logs, exist_ok=True)
    current_datetime = datetime.utcnow()
    formatted_datetime = current_datetime.strftime('%Y-%m-%d_%H:%M')
    script_name = sys.argv[0].split("/")[-1].split(".")[0]
    log_file = f'{logs}{formatted_datetime.split("_")[0]}_{script_name}.log'
    sys.stdout = Logger(logfile=log_file)
    print(f"\nRun date, UTC: {datetime.utcnow()}")
    print(f"Run settings: {sys.argv[0]} {' '.join(f'--{k} {v}' for k, v in vars(args).items())}")


class Logger(object):
    def __init__(self, logfile=None):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")  # overwrite, don't "a" append

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


ANNOTATORS = {'ende': {
    'adrian': 'nexor',
    'david': 'zorin',
    'houda': 'lyric',
    'maria_de': 'beaut'},
    'deen': {
        'laura': 'velin',
        'ella': 'lumie',
        'maria_en': 'sorry'}
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--empty', help="", default='assess_in/sorry.tsv')
    parser.add_argument('--anno', help="", default='assess_out/')
    parser.add_argument('--res', default='res/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.res])

    for subdir in ['proxy/', 'real/']:  #  'proxy/', real/
        save_as = f'{args.res}deen_{subdir[:-1]}_human_scores.tsv'
        if subdir == 'proxy/':
            if os.path.exists(save_as):
                df = pd.read_csv(save_as, sep='\t')
                print(f'Proxy data exists.')
            else:
                names = list(ANNOTATORS['deen'].values())
                df = generate_proxy(args.empty, raters=names)
            print(df.head())
            print(df.shape)
            df.to_csv(save_as, sep='\t', index=False)
        else:
            if os.path.exists(args.anno + subdir):

                for lpair in ['deen', 'ende']:  #
                    collector = []
                    names_d = ANNOTATORS[lpair]
                    for rater in list(names_d.values()):
                        this_rater = [f for f in os.listdir(args.anno + subdir) if rater in f][0]
                        if this_rater:
                            # this rater is special. Their table includes automatic scores.
                            if rater == 'sorry' or rater == 'beaut':
                                df = pd.read_csv(args.anno + subdir + this_rater, sep='\t',
                                                 usecols=['base_seg_id', 'source', 'ht', 'auto_rank', 'score',
                                                          "bleu", "chrf", "ter", "comet_ref", "comet_qe"]).set_index('base_seg_id')
                            else:
                                df = pd.read_csv(args.anno + subdir + this_rater, sep='\t',
                                                 usecols=['base_seg_id', 'score']).set_index('base_seg_id')
                            df = df.rename(columns={'score': rater})

                            collector.append(df)
                        else:
                            continue

                    ol_scores = pd.concat(collector, axis=1).reset_index()
                    ol_scores.insert(ol_scores.shape[1], 'lpair', lpair)
                    ol_scores.to_csv(save_as.replace('deen_', f'{lpair}_'), sep='\t', index=False)
                    print(ol_scores.head())
                    print(f'Collected scores from {len(collector)} annotators.')

            else:
                print(f'Put files with real annotations into {args.anno + subdir}.')

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
