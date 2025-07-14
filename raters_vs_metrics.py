"""
02 Jul 2025

USAGE:
python3 raters_vs_metrics.py

"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore


def decode_raters(my_df, my_map=None):
    reversed_map = {v: k for k, v in my_map.items()}
    my_df.rename(columns=reversed_map, inplace=True)
    reversed_map_scaled = {f"{v}_scaled": f"{k}_scaled" for k, v in my_map.items()}
    my_df.rename(columns=reversed_map_scaled, inplace=True)
    good_raters = [rater for rater in list(reversed_map.values()) if rater in my_df.columns.tolist()]

    return my_df, good_raters


def trend_and_fit(normed_df, numeric_feats, response, save_pic,
                  my_lpair=None, show=True,
                  feature_color_map=None,
                  trend='spearman'):
    results = []
    for feat in numeric_feats:
        if trend == "spearman":
            coef, pval = spearmanr(normed_df[feat], normed_df[response])
        else:
            coef, pval = pearsonr(normed_df[feat], normed_df[response])
        signif = "yes" if pval < 0.05 else "no"
        direction = "pos" if coef > 0 else "neg"
        results.append((feat, coef, signif, direction))

    # Create summary DataFrame
    summary_df = pd.DataFrame(results, columns=["auto_metric", "coef", "signif", "directionality"])

    # Merge data for plotting
    plot_df = summary_df.copy()
    plot_df = plot_df.sort_values("coef", ascending=False)
    plot_df["label"] = plot_df.apply(
        lambda row: f"{row['auto_metric']} ({row['coef']:.2f}{'*' if row['signif'] == 'yes' else ''})", axis=1
    )

    plot_df["color"] = plot_df["auto_metric"].map(feature_color_map)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, row in plot_df.iterrows():
        # this is linear regression best fit line and the confidence interval
        sns.regplot(
            x=normed_df[row["auto_metric"]],
            y=normed_df[response],
            scatter_kws={'alpha': 0.3, 'color': row["color"]},
            line_kws={'color': row["color"]},
            scatter=False,
            label=row["label"],
            ci=95,
            ax=ax
        )

    if trend == "spearman":
        ax.legend(title="Auto metrics and trend (Spearman *=signif)",
                  loc='best', fontsize=18, title_fontsize=20)
    else:
        ax.legend(title="Auto metrics (Pearson *=signif)", loc='best', fontsize=18, title_fontsize=20)
    ax.set_xlabel("Z-transformed auto metric values", fontsize=20)
    if 'scaled' in save_pic:
        ax.set_ylabel(f"Z-transformed human scores: {response}", fontsize=20)
    else:
        ax.set_ylabel(f"Human scores: {response}", fontsize=18)
    ax.set_title(f"{my_lpair.upper()}: Univariate linear regression (best fit)", fontsize=20)
    ax.tick_params(axis='both', labelsize=16)
    sns.despine()
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(save_pic, dpi=300)
    if show:
        plt.show()
    plt.close()

    return summary_df

# same as zcore from scipy.stats zcore
def standardise_scores(scores_list):
    scores_2d = np.array(scores_list).reshape(-1, 1)  # this is done to fit the requirements of the scaling function
    scaler = StandardScaler()
    normed_scores = scaler.fit_transform(scores_2d)
    normed_scores = normed_scores.squeeze()  # Shape becomes (131,)

    return normed_scores


# same as zcore from scipy.stats zcore
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
    parser.add_argument('--anno', help="the dir with the post processed annotations", default='res/')
    parser.add_argument('--decode_raters', help="", default=True)
    parser.add_argument('--no_scaling', action='store_true', help='pass this flag to run WITHOUT z-transform')
    parser.add_argument('--pics', default='pics/')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.pics])

    collector = []
    for lpair in ['deen', 'ende']:  # 'deen', 'ende'
        df = pd.read_csv(f'{args.anno}{lpair}_{SCORE}_human_scores.tsv', sep='\t')
        # drop raters with constant scores (None)
        df = df.dropna(axis=1, how='all')

        if args.decode_raters:
            df, names = decode_raters(df, my_map=ANNOTATORS[lpair])
        else:
            names = [rater for rater in list(ANNOTATORS[lpair].values()) if rater in df.columns.tolist()]

        print(f"No scaling?: {args.no_scaling}")

        scaler = StandardScaler()
        if not args.no_scaling:
            for nr, annotator in enumerate(names):
                # z-transform: The scores now have a mean of 0 and a standard deviation of 1.
                # Needed to take into account individual human biases
                # this_rater_scores = df[annotator].values
                # df[f'{annotator}_scaled'] = standardise_scores(this_rater_scores)

                # this implementation handles Nones
                df[f'{annotator}_scaled'] = zscore(df[annotator], nan_policy='omit')
        names_scaled = [f"{rater}_scaled" for rater in names]
        if len(names) <= 1:
            pass
        else:
            df['mean_raters'] = df[names].mean(axis=1)
            df['mean_raters_scaled'] = df[names_scaled].mean(axis=1)

        # aggregate auto
        df['ter_inv'] = 100 - df['ter']  # invert "lower, better" to "higher, better"
        metrics = ['bleu', 'chrf', 'ter_inv', 'comet_ref', 'comet_qe']

        # z-transform to account for scale differences between autometrics
        df_normed = pd.DataFrame(scaler.fit_transform(df[metrics]), columns=metrics)
        df['mean_normed_metrics'] = df_normed.mean(axis=1)

        # print(df.head())
        # print(df.shape)

        # print(df.columns.tolist())
        ol_auto = metrics + ['mean_normed_metrics']
        # Create a discrete colormap with as many colors as features
        cmap = plt.get_cmap("viridis", len(ol_auto))  # diff_cols
        my_coloured_feats = {feat: mcolors.to_hex(cmap(i)) for i, feat in enumerate(ol_auto)}  # diff_cols
        if len(names) <= 1:
            humans = names + names_scaled
        else:
            humans = ['mean_raters', 'mean_raters_scaled'] + names + names_scaled

        for manual_scores in humans:
            save_pic = f"{args.pics}{lpair}_{manual_scores}_vs_auto_assessment.png"

            coef_res = trend_and_fit(df, numeric_feats=ol_auto,
                                     response=manual_scores, save_pic=save_pic,
                                     my_lpair=lpair, show=args.verbose,
                                     feature_color_map=my_coloured_feats,
                                     trend='spearman')

            print(coef_res)

        # vertical or short lines indicate no variance in predictor (x-axis) -> no meaningful regression
        print(lpair)
        print(df.describe())

    end = time.time()
    print(f'\nTotal time: {((end - start) / 60):.2f} min')
