"""
01 Jul 2024

This script uses a variant of COMET calculation using evaluation utility
Requirements:
pip install --upgrade pip
pip install unbabel-comet
huggingface-cli login

USAGE:
For COMET metric, there are two types of models (1) without reference (2) with prof_translation as reference
(1) Unbabel/wmt22-comet-da --comet_model
(2) Unbabel/wmt22-cometkiwi-da --qe

Expected input: a multiparallel table data/human_deepl.tsv with 'src_doc_id', 'lpair', 'source', 'ht' and 'mt' columns
(you would probably have a unique segment id column, too. Here: base_seg_id)
Run command:
python3 automatic_metrics.py

"""

import sys

import pandas as pd
import argparse
import os
import time
from datetime import datetime
import sacrebleu
from comet import download_model, load_from_checkpoint


def get_parallel_lists(my_df, src_col, tgt_col, ref_col):
    return (
        my_df[src_col].tolist(),
        my_df[tgt_col].tolist(),
        my_df[ref_col].tolist()
    )

# outdated since 24Jul 2024!
# def comet_compute(s=None, t=None, r=None, my_metric=None, comet_model=None, gpus=0):
#     this_metric = evaluate.load(my_metric, comet_model)
#     # unwanted arguments are simply ignored
#     results = this_metric.compute(predictions=t, references=r, sources=s, gpus=gpus, progress_bar=True)
#     score_list = results["scores"]
#     return score_list


def comet_compute(src=None, tgt=None, ref=None, comet_model=None, gpus=0):
    model_path = download_model(comet_model)
    model = load_from_checkpoint(model_path)

    data = [
        {"src": s, "mt": t, "ref": r}
        for s, t, r in zip(src, tgt, ref)
    ]
    model_output = model.predict(data, batch_size=8, gpus=gpus)
    scores = model_output.scores

    return scores


def sacrebleu_compute(mt, ref, metrics=None):
    scores = {}
    if "bleu" in metrics:
        scores["bleu"] = sacrebleu.sentence_bleu(mt, [ref]).score
    if "chrf" in metrics:
        scores["chrf"] = sacrebleu.sentence_chrf(mt, [ref]).score
    if "ter" in metrics:
        scores["ter"] = sacrebleu.sentence_ter(mt, [ref]).score
    return [scores[m] for m in metrics]


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


MY_METRICS = {'sacrebleu': ['bleu', 'chrf', 'ter'],
              'comet': ['ref', 'qe']}

COMET_MODELS = {'ref': 'Unbabel/wmt22-comet-da',     # Unbabel/wmt22-cometkiwi-da, zwhe99/wmt21-comet-qe-da
                'qe': 'Unbabel/wmt22-cometkiwi-da'}  # 'wmt20-comet-da', 'wmt21-comet-da', 'Unbabel/wmt22-comet-da'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate autometrics discussed in QHMT seminar')
    parser.add_argument('--intab', help="", default='data/human_deepl.tsv')
    parser.add_argument('--res', default='res/')
    parser.add_argument('--logs', default='logs/')

    args = parser.parse_args()
    start = time.time()

    make_dirs(logs=args.logs, make_them=[args.res])

    multi_par = pd.read_csv(args.intab, sep='\t')
    collector = []
    for lpair in multi_par['lpair'].unique():  # ['ende', 'deen']:
        lpair_multi_par = multi_par[multi_par['lpair'] == lpair]
        auto_anno = args.intab.replace('data/', 'res/').replace('.tsv', f'_auto_{lpair}.tsv')
        if os.path.exists(auto_anno):
            multi_auto = pd.read_csv(auto_anno, sep='\t')
            print(f'{lpair.upper()}: Loading autometrics annotations. Delete {auto_anno} to run anew.')
        else:
            print(f'\n{lpair.upper()}: Calculating autometrics anew\n')
            multi_auto = lpair_multi_par.copy()
            sacrebleus = MY_METRICS["sacrebleu"]
            multi_auto[sacrebleus] = multi_auto.apply(
                lambda row: sacrebleu_compute(row['mt'], row['ht'], metrics=sacrebleus),
                axis=1, result_type="expand"
            )

            src, tgt, ref = get_parallel_lists(my_df=multi_auto, src_col='source', tgt_col='mt', ref_col='ht')

            comets = MY_METRICS['comet']
            for metric in comets:
                model = COMET_MODELS[metric]
                comet_scores = comet_compute(src=src, tgt=tgt, ref=ref, comet_model=model, gpus=0)
                print(len(comet_scores), len(src), len(tgt))
                assert len(comet_scores) == len(src) == len(tgt), 'Huston, we have problems!'
                multi_auto.insert(multi_auto.shape[1], f'comet_{metric}', comet_scores)

            multi_auto.to_csv(auto_anno, sep='\t', index=False)

        # ranking
        agg_dict = {col: 'mean' for col in ["bleu", "chrf", "ter", "comet_ref", "comet_qe"]}
        agg_dict["source"] = "count"  # Add sentence count
        doc_scores = multi_auto.groupby("src_doc_id").agg(agg_dict).rename(columns={"source": "n_sents"}).reset_index()
        # doc_scores = multi_auto.groupby("src_doc_id")[["bleu", "chrf", "ter", "comet_ref", "comet_qe"]].mean().reset_index()
        for metric in ["bleu", "chrf", "comet_ref", "comet_qe"]:
            doc_scores[f"{metric}_rank"] = doc_scores[metric].rank(ascending=False, method="min")

        # For TER, lower is better, rank in ascending order
        doc_scores["ter_rank"] = doc_scores["ter"].rank(ascending=True, method="min")

        # identify the worst doc by summing ranks
        doc_scores["total_rank"] = doc_scores[["bleu_rank", "chrf_rank", "ter_rank",
                                               "comet_ref_rank", "comet_qe_rank"]].sum(axis=1)
        doc_scores_sorted = doc_scores.sort_values("total_rank")  # rank 1 is best
        doc_scores_sorted.insert(0, 'lpair', lpair)
        collector.append(doc_scores_sorted)

    final = pd.concat(collector, axis=0)
    print(final)
    final.to_csv(f'{args.res}ranked_docs.tsv', sep='\t', index=False)

    endtime_tot = time.time()
    print(f'\nTotal time: {((endtime_tot - start) / 60):.2f} min')
