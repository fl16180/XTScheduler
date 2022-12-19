import os
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

from .table import format_df_table


class TrackViewer:
    def __init__(self, log_dir, fmt_string='log_DATASET_MODEL_ID.csv'):
        self.source_dir = Path(log_dir)
        self.fmt_string = fmt_string
        self.track_params = {}

    def set_datasets(self, datasets):
        self.track_params['DATASET'] = datasets

    def set_models(self, models):
        self.track_params['MODEL'] = models

    def set_run_ids(self, ids):
        self.track_params['ID'] = ids

    def set_param(self, name, options):
        self.track_params[name] = options

    def set_param_dict(self, param_dict):
        for key, val in param_dict.items():
            self.track_params[key] = val

    def set_metrics(self, epoch_metrics=[], global_metrics=[]):
        self.epoch_metrics = epoch_metrics
        self.global_metrics = global_metrics

    # def _gather_tracks(self):
    #     expected_tracks = []
    #     for values in itertools.product(*self.track_params.values()):
    #         combo = dict(zip(self.track_params.keys(), values))

    #         log_name = self.fmt_string
    #         for key, val in combo.items():
    #             log_name = log_name.replace(key, str(val))

    #         expected_tracks.append(log_name)
    #     return expected_tracks

    def _gather_tracks(self):
        for values in itertools.product(*self.track_params.values()):
            combo = dict(zip(self.track_params.keys(), values))
            log_name = self.fmt_string
            for key, val in combo.items():
                log_name = log_name.replace(f'_{key}', f'_{str(val)}')
            yield log_name, combo

    def _validate_tracks(self, expected_tracks):
        missing_tracks = []
        for log_name, _ in expected_tracks:
            if not os.path.exists(self.source_dir / log_name):
                missing_tracks.append(log_name)
        if len(missing_tracks) > 0:
            print('Missing logs: ', missing_tracks)
        else:
            print('All logs present')
        return missing_tracks

    def _load_global_metrics(self, log_path):
        df = pd.read_csv(log_path)
        
        # check that it is valid global metric (all rows same)
        for col in self.global_metrics:
            assert len(df[col].unique()) == 1

        metrics = df[self.global_metrics].iloc[0:1].copy()
        return metrics

    def _load_epoch_metrics(self, log_path):
        df = pd.read_csv(log_path)
        metrics = df[self.epoch_metrics].copy()
        return metrics

    # def _load_track_as_unit(self, log_path):
    #     df = pd.read_csv(log_path)
    #     from pdb import set_trace; set_trace()
    #     df = df[self.epoch_metrics + self.global_metrics].mean()

    def _load_track(self, log_path):
        df = pd.read_csv(log_path)
        return df[self.epoch_metrics + self.global_metrics]

    def compile_tracks(self):
        expected_tracks = self._gather_tracks()
        missing_tracks = self._validate_tracks(expected_tracks)

        expected_tracks = self._gather_tracks()
        results = []
        for track, param_combo in expected_tracks:
            if track in missing_tracks:
                continue

            metrics = self._load_track(self.source_dir / track)
            # metrics = self._load_track_as_unit(self.source_dir / track)
            # metrics = self._load_global_metrics(self.source_dir / track)
            for key, val in param_combo.items():
                metrics[key] = val
            results.append(metrics)

        results = pd.concat(results, axis=0)
        return results

    def _run_aggregation(self, results, agg_over, agg_type):
        groupby_vars = [x for x in self.track_params if x not in agg_over]
        print('Category variables: ', groupby_vars)

        if agg_type == 'mean':
            df = results.groupby(groupby_vars, as_index=False).mean()
        elif agg_type == 'median':
            df = results.groupby(groupby_vars, as_index=False).median()
        elif agg_type == 'std':
            df = results.groupby(groupby_vars, as_index=False).std()
        return df

    def compile_table(self, agg_over=['ID'], agg_type='mean', compute_std=False,
                      main_decimals=3, std_decimals=2, std_fmt='sub', bold_values='max'):
        expected_tracks = self._gather_tracks()
        missing_tracks = self._validate_tracks(expected_tracks)

        results = []
        for track in expected_tracks:
            if track in missing_tracks:
                continue

            metrics = self._load_global_metrics(self.source_dir / track)
            for key, val in self.track_params.items():
                metrics[key] = val
            results.append(metrics)

        results = pd.concat(results, axis=0)
        
        results = self._run_aggregation(results, agg_over, agg_type)
        results = results.round(main_decimals)
        results = format_df_table(results, bold_cols=self.global_metrics, bold_values=bold_values)

        if compute_std:
            spreads = self._run_aggregation(results, agg_over, 'std')
            spreads = spreads.round(std_decimals)

    # def compile_steps(self, metric, )


def div_learn():

    logger = DivLogAnalyzer(
        source_dir = './results/classify',
        fmt_string='log_div_DATASET_MODEL_ID.csv'
    )

    logger.set_datasets(['iris', 'car', 'wine', 'balance-scale', 'transfusion', 'abalone'])
    logger.set_models(['nbd', 'deep-div', 'pbdl', 'euclidean', 'mahalanobis', 'deepnorm', 'widenorm'])
    logger.set_run_ids(list(range(0, 20)))

    results = logger.compile_runs()

    means, stds = logger.aggregate(results)
    means = means.round(3)
    stds = stds.round(2)

    means = means.applymap(lambda x: str(x))
    stds = stds.applymap(lambda x: str(x))

    def combine_fn(x):
        if x.name in ['dataset', 'model']:
            return x
        else:
            return r'$' + x + r'_{\ ' + stds[x.name] + r'}$'

    df = means.apply(combine_fn)

    df['model'] = df['model'].replace({
        'deep-div': 'Deep-div',
        'euclidean': 'Euclidean',
        'mahalanobis': 'Mahalanobis',
        'nbd': 'NBD',
        'pbdl': 'PBDL'
    })

    df.set_index(['dataset', 'model'], inplace=True)
    output = df.to_latex(multirow=True, escape=False)
    print(output)


def mixture():

    logger = DivLogAnalyzer(
        source_dir = './results/mixture',
        fmt_string='log_dist_DATASET_MODEL_EMBED_mix_MIX_norm_NORM_ID.csv'
    )

    logger.set_datasets(['gaussian', 'multinomial', 'exponential'])
    logger.set_models(['nbd', 'deep-div', 'pbdl', 'euclidean', 'mahalanobis'])
    logger.set_run_ids(list(range(0, 10)))
    logger.set_param('EMBED', ['none'])
    logger.set_param('MIX', ['False'])
    logger.set_param('NORM', ['False'])

    results = logger.compile_runs()
    results = results[results.embed == 'none']
    results.drop('id', axis=1, inplace=True)

    df = results.groupby(['dataset', 'model', 'embed', 'mix'], as_index=False).mean()
    df.drop('embed', axis=1, inplace=True)
    df.drop('mix', axis=1, inplace=True)
    df = df.pivot(index=['model'], columns=['dataset'])
    means = df.swaplevel(axis=1).sort_index(axis=1, level=0).round(3)

    df = results.groupby(['dataset', 'model', 'embed', 'mix'], as_index=False).std()
    df.drop('embed', axis=1, inplace=True)
    df.drop('mix', axis=1, inplace=True)
    df = df.pivot(index=['model'], columns=['dataset'])
    stds = df.swaplevel(axis=1).sort_index(axis=1, level=0).round(2)

    means = means.applymap(lambda x: str(x))
    stds = stds.applymap(lambda x: str(x))

    def combine_fn(x):
        if x.name in ['model']:
            return x
        else:
            return r'$' + x + r'_{\ ' + stds[x.name] + r'}$'

    df = means.apply(combine_fn)
    output = df.to_latex(multirow=True, escape=False)
    print(output)


def deep_div():

    logger = DivLogAnalyzer(
        source_dir = './results/deep_classify',
        fmt_string='log_div_DATASET_MODEL_norm_False_ID.csv'
    )
    logger.set_datasets(['fmnist', 'cifar10', 'stl10', 'svhn'])
    logger.set_models(['nbd', 'deep-div', 'euclidean'])
    logger.set_run_ids(list(range(0, 3)))

    results = logger.compile_runs()
    res = results.groupby(['dataset', 'model']).mean().round(3)
    print(res[['knn@1', 'knn@5', 'prec@10', 'map@10']].to_latex())


def text():
    logger = DivLogAnalyzer(
        source_dir = './results/text',
        fmt_string='log_DATASET_MODEL_ID.csv'
    )
    logger.set_datasets(['text'])
    logger.set_models(['deep-div', 'euclidean', 'mahalanobis', 'nbd'])
    logger.set_run_ids(list(range(0, 10)))

    results = logger.compile_runs()

    from pdb import set_trace; set_trace()

if __name__ == '__main__':
    # mixture()
    # deep_div()
    # div_learn()

    text()