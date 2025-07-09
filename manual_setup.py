"""
01 Jul 2024
Create manual annotation task
***
Annotation Guidelines: Translation Quality Assessment using Error Span Annotation (ESA)
Read the source text and evaluate the translation quality of each sentence in context using an ordinal 1 to 10 scale, where:
1 = Unacceptable, misrepresenting the source content
10 = Perfect, publishable translation
If you downgrade a translated sentence, do the following:
Highlight the faulty span(s) in the target sentence.
OPTIONALLY: For each identified error, use a new column to assign the error severity (minor, major) in the order the errors appear, using these categories.
For sentences where there are more than five major errors, enter 0 score (equal to non-translation).
If information is missing from the translation, highlight the [MISSING] placeholder and mark its severity.
***

For reliable inter-annotator score (Krippendorff's αlpha), we need n > 30

python3 manual_setup.py

"""

import sys

import pandas as pd
import argparse
import os
import time
from datetime import datetime
import sacrebleu
from comet import download_model, load_from_checkpoint


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


# documents with the highest mean for segment-level translation task difficulty (worst) and the lowest seg-TTD
MY_CONTRASTS = {'ende': {'best': 'ORG_WR_EN_DE_003752', 'worst': 'ORG_WR_EN_DE_004321'},
                'deen': {'best': 'ORG_WR_DE_EN_018529', 'worst': 'ORG_WR_DE_EN_006043'}}

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
    parser.add_argument('--intab', help="", default='data/human_deepl.tsv')
    parser.add_argument('--res', default='assess_in/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.res])

    multi_par = pd.read_csv(args.intab, sep='\t').set_index('base_seg_id')
    diff_prob = pd.read_csv('data/seg_difficulty_scores.tsv', sep='\t',
                            usecols=['base_seg_id', 'prob_target', 'mean_diff',
                                     'corr_weighted_diff', 'pca_diff']).set_index('base_seg_id')
    merged = pd.concat([multi_par, diff_prob], axis=1)
    merged = merged.dropna().reset_index()
    print(merged.shape)
    print(merged.head())
    for lpair in merged['lpair'].unique():  # ['ende', 'deen']::
        collector = []
        lpair_merged = merged[merged['lpair'] == lpair]
        for auto_rank in ['best', 'worst']:
            doc_id = MY_CONTRASTS[lpair][auto_rank]
            print(doc_id)
            segs = lpair_merged[lpair_merged['src_doc_id'] == doc_id].reset_index()

            n = 5
            segs.insert(segs.shape[1], 'auto_rank', auto_rank)

            segs = segs.copy()
            segs['target'] = segs['mt']
            # overwriting
            segs.loc[:n - 1, 'target'] = segs.loc[:n - 1, 'ht']
            segs.loc[:n - 1, 'auto_rank'] = 'human'
            segs.loc[:, 'target'] = segs['target'].astype(str) + " [MISSING]"
            segs.insert(segs.shape[1] - 1, 'score', None)
            segs.insert(segs.shape[1] - 1, 'severity', None)

            # print(segs[['base_seg_id', 'ht', 'target', 'auto_rank']].head())
            # input()
            collector.append(segs)

        this_dir = pd.concat(collector, axis=0)

        print(this_dir.columns.tolist())

        for nick in ANNOTATORS[lpair].values():
            if nick == 'sorry' or nick == 'beaut':

                marias = this_dir[['base_seg_id', 'source', 'target', 'score', 'severity', 'ht', 'auto_rank']].set_index(
                    'base_seg_id')
                seg_auto_qua = pd.read_csv(f'res/human_deepl_auto_{lpair}.tsv',
                                           usecols=['base_seg_id', "bleu", "chrf", "ter", "comet_ref", "comet_qe"],
                                           sep='\t').set_index('base_seg_id')
                anno_out = pd.concat([marias, seg_auto_qua], axis=1).reset_index()
                anno_out = anno_out.dropna(subset=['source'])

            else:
                anno_out = this_dir[['base_seg_id', 'source', 'target', 'score', 'severity']]
            anno_out.to_csv(f'{args.res}{nick}.tsv', sep='\t', index=False)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
