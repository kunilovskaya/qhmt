"""
02 Jul 2025

USAGE:
python3 interrater_agreement.py

"""

import argparse
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import numpy as np

import krippendorff
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_plottable(my_scores, my_map=None):
    # re-format the data for heatmap
    df1 = my_scores.copy()
    df1 = df1.stack().reset_index()

    df1.columns = ['ID', 'rater', 'group']

    annotation2 = []
    # get all the individual annotators
    for name in df1['rater'].unique():
        labels = []
        # iterate over all the rows in the dataframe
        for x, row in df1[df1['rater'] == name].iterrows():
            labels.append(row["group"])
        annotation2.append((name, labels))  # [(rater1, [score1, score2, score3, ...])]
    # print(annotation2[:3])
    # print(len(annotation2))

    # get a heat_map based on Krippendorff's scores
    df_heat2 = pd.DataFrame(index=df1['rater'].unique(), columns=df1['rater'].unique(), dtype=np.float32)

    for name1, cl1 in annotation2:
        for name2, cl2 in annotation2:
            df_heat2.at[name1, name2] = np.float32(
                krippendorff.alpha([cl1, cl2], level_of_measurement=args.score_type)
            )

    reversed_map = {v: k for k, v in my_map.items()}
    df_heat2.rename(index=reversed_map, columns=reversed_map, inplace=True)
    print(df_heat2.head())

    return df_heat2


def corr_heatmap(plot_df, save_pic=None, show=None):
    sns.set_style("whitegrid")
    sns.set_context('paper')
    sns.set(rc={'figure.figsize': (11.7, 8.27)})

    g = sns.heatmap(plot_df, annot=True, fmt="f", cmap="YlGnBu")
    g.set_yticklabels(g.get_yticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig(save_pic, dpi=300)
    if show:
        plt.show()
    plt.close()


def standardise_scores(scores_list):
    scores_2d = np.array(scores_list).reshape(-1, 1)  # this is done to fit the requirements of the scaling function
    scaler = StandardScaler()
    normed_scores = scaler.fit_transform(scores_2d)
    normed_scores = normed_scores.squeeze()  # Shape becomes (131,)

    return normed_scores


# same as
def standardise_scores_manual(scores_list):
    # Calculate mean and standard deviation for each rater
    mean_rater, std_rater = np.mean(scores_list), np.std(scores_list)
    # Standardize each rater's scores
    normed_scores = (scores_list - mean_rater) / std_rater

    return normed_scores


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
    args = sys.argv[1:]
    print(f"Run settings: {sys.argv[0]} {' '.join(i for i in args)}")


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


SCORE = 'real'  # real, proxy
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', help="the dir with the post processed annotations", default='res/')
    parser.add_argument('--no_scaling', action='store_true', help='pass this flag to run WITHOUT z-transform')
    parser.add_argument('--score_type', choices=['nominal', 'interval', 'ordinal'], default='ordinal',
                        help="A 1–10 translation rating scale is typically considered ordinal data.")
    parser.add_argument('--pics', default='pics/')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.pics])

    collector = []
    for lpair in ['deen']:  # 'ende',  we don't have real annotations for more than one rater in ende direction
        print(lpair)
        res = defaultdict()
        raters_cols = list(ANNOTATORS[lpair].values())
        try:
            my_data = [f for f in os.listdir(args.indir) if SCORE in f and lpair in f][0]
        except IndexError:
            continue
        df = pd.read_csv(args.indir + my_data, sep='\t').set_index('base_seg_id')
        print(df.head())

        human_scores = df[raters_cols]
        auto_scores = df[["bleu", "chrf", "comet_ref", "comet_qe"]]  # "ter", 
        # calculate Krippendorff's alpha with a dedicated python module
        for meth, score_df in zip(['raters', 'metrics'], [human_scores, auto_scores]):
            # reformat data as required by the module: create an empty df
            scores = np.zeros((len(score_df.columns), score_df.shape[0]))  # this does not accept nominal values
            # fill in the empty df
            for nr, annotator in enumerate(score_df.columns):
                this_rater_scores = score_df[annotator].values
                if not args.no_scaling:
                    # print('\n*** StandardScaling scores, assuming symmetric distribution: '
                    #       'to level out human bias and differences in scales for auto metrics'
                    #       'The scores now have a mean of 0 and a standard deviation of 1. ')
                    this_rater_scores = standardise_scores_manual(this_rater_scores)

                scores[nr, :] = this_rater_scores

            print(f'{meth.capitalize()}:', scores.shape[0])
            print('Instances:', scores.shape[1])

            agreement = krippendorff.alpha(scores, level_of_measurement=args.score_type)

            res[meth] = agreement

            pairwise_res = get_plottable(score_df, my_map=ANNOTATORS[lpair])
            
            pic_name = f'{args.pics}{lpair}_corr_{meth}.png'
            corr_heatmap(pairwise_res, save_pic=pic_name, show=args.verbose)

        res_df = pd.DataFrame([res])
        res_df.insert(0, 'lpair', lpair)
        res_df.index = res_df.index.map(lambda x: f"Krippendorff's alpha ({args.score_type})" if x == 0 else x)

        collector.append(res_df)

    ol_res = pd.concat(collector, axis=0)
    print(ol_res)

    end = time.time()
    print(f'\nTotal time: {((end - start) / 60):.2f} min')
