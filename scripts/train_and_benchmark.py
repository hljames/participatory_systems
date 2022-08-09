"""
Train models, benchmark on datasets, and save results to results file
"""

import dill
import numpy as np
import os
import pandas as pd
import psutil

from psc.auditor import PersonalizationAuditor
from psc.utils import print_log, METRIC_NAMES
from psc.groups import MISSING_VALUE
from psc.data import BinaryClassificationDataset
from psc.participatory import FlatSystem, SequentialSystem
from psc.paths import get_processed_data_file, get_training_results_file
from psc.training import train_sklearn_linear_model
from psc.utils import powerset, raw_gain, MODEL_TYPES


def train_models(settings):
    print(settings)
    # load dataset
    data_file_processed = get_processed_data_file(settings['data_name'], rebalancing_type=settings['rebalancing_type'])
    data = BinaryClassificationDataset.load(file=data_file_processed)

    # create a lambda function to train classifier from data
    if settings['method_name'] in ('logreg'):
        def train_model(X, G, y, settings, normalize_variables):
            return train_sklearn_linear_model(X, G, y, method_name=settings['method_name'], settings=settings,
                                              normalize_variables=normalize_variables)
    else:
        raise NotImplementedError()

    # split dataset into folds
    data.split(fold_id=settings['fold_id'],
               fold_num_validation=settings['fold_num_validation'],
               fold_num_test=settings['fold_num_test'])

    # fit generic model
    generic_model = train_model(X=data.training.X, G=None, y=data.training.y, settings=settings,
                                normalize_variables=False)

    all_models = {}

    assert all([model in MODEL_TYPES for model in settings['models']])
    if 'onehot' in settings['models']:
        curr_settings = settings.copy()
        curr_settings['encoding_type'] = 'onehot'
        h = train_model(X=data.training.X, G=data.training.G, y=data.training.y, settings=curr_settings,
                        normalize_variables=False)
        all_models['onehot'] = h

    if 'intersectional' in settings['models']:
        curr_settings = settings.copy()
        curr_settings['encoding_type'] = 'intersectional'
        h = train_model(X=data.training.X, G=data.training.G, y=data.training.y, settings=curr_settings,
                        normalize_variables=False)
        all_models['intersectional'] = h

    if 'onehot_impute' in settings['models'] or 'intersectional_impute' in settings['models']:
        curr_settings = settings.copy()
        curr_settings['impute_group_attributes'] = True
        if 'onehot_impute' in settings['models']:
            curr_settings['encoding_type'] = 'onehot'
            h = train_model(X=data.training.X, G=data.training.G, y=data.training.y, settings=curr_settings,
                            normalize_variables=False)
            all_models['onehot_impute'] = h
        if 'intersectional' in settings['models']:
            curr_settings['encoding_type'] = 'intersectional'
            h = train_model(X=data.training.X, G=data.training.G, y=data.training.y, settings=curr_settings,
                            normalize_variables=False)
            all_models['intersectional_impute'] = h

    if 'flat' in settings['models'] or 'sequential' in settings['models']:
        X_train = data.training.X
        y_train = data.training.y
        G_train = data.training.G.reset_index(drop=True)
        candidate_models = []
        for encoding_type in ['onehot', 'intersectional']:
            for name_subset in powerset(data.group_attributes.names, min_size=1):
                curr_settings = settings.copy()
                curr_settings['encoding_type'] = encoding_type
                G_name_subset = G_train[list(name_subset)]
                curr_settings['training_groups'] = data.group_encoder.groups
                # add candidate models trained on subsets of group attributes over all groups
                h = train_sklearn_linear_model(X_train, G_name_subset, y_train,
                                               method_name=curr_settings['method_name'],
                                               settings=curr_settings, normalize_variables=False)

                candidate_models.append(h)
                h_labels = h._group_encoder.labels
                # Add decoupled models
                for i, group_label in enumerate(data.group_encoder.groups):
                    curr_settings['training_groups'] = [group_label]
                    q = ' & '.join(
                        [f"""{n} == '{v}'""" for n, v in zip(G_train, group_label)])
                    # print('train with ', G_name_subset.columns, ' on ', q)
                    idx = G_train.query(q).index.tolist() if q else G_train.index.tolist()
                    if not len(idx):
                        print('no samples found for ', q)
                        continue
                    y_train_idx = y_train[idx]
                    if np.all(y_train_idx == 1.):
                        y_train_idx[-1] = -1.
                    elif np.all(y_train_idx == -1.):
                        y_train_idx[-1] = 1.
                    h = train_sklearn_linear_model(X_train[idx], G_name_subset.iloc[idx], y_train_idx,
                                                   method_name=curr_settings['method_name'],
                                                   settings={**curr_settings, **{'labels': h_labels}},
                                                   normalize_variables=False)
                    candidate_models.append(h)

        if 'flat' in settings['models']:
            p_flat = FlatSystem(data, generic_model, assignment_metric=settings['assignment_metric'],
                                assignment_sample=settings['assignment_sample'])
            p_flat.update_assignments(candidate_models)
            all_models['flat'] = p_flat

        if 'sequential' in settings['models']:
            p_seq = SequentialSystem(data, generic_model, assignment_metric=settings['assignment_metric'],
                                     assignment_sample=settings['assignment_sample'])
            p_seq.update_assignments(candidate_models)
            all_models['sequential'] = p_seq

    if 'participatory_simple' in settings['models']:
        curr_settings = settings.copy()
        curr_settings['encoding_type'] = 'onehot'
        h = train_model(X=data.training.X, G=data.training.G, y=data.training.y, settings=curr_settings,
                        normalize_variables=False)
        candidate_models = [h, generic_model]
        p_simple = FlatSystem(data, generic_model, assignment_metric=settings['assignment_metric'],
                              assignment_sample=settings['assignment_sample'])
        p_simple.update_assignments(candidate_models)
        all_models['participatory_simple'] = p_simple
    if 'decoupled' in settings['models']:
        curr_settings = settings.copy()
        curr_settings['encoding_type'] = 'onehot'
        decoupled = FlatSystem(data, generic_model, assignment_metric=curr_settings['assignment_metric'],
                               assignment_sample=curr_settings['assignment_sample'])
        X_train = data.training.X
        y_train = data.training.y
        G_train = data.training.G.reset_index(drop=True)
        candidate_models = []
        h_labels = {c: list(np.unique(data.G[c])) for c in data.G.columns}
        for i, group_label in enumerate(data.group_encoder.groups):
            curr_settings['training_groups'] = [group_label]
            q = ' & '.join(
                [f"""{n} == '{v}'""" for n, v in zip(G_train, group_label)])
            # print('train with ', G_name_subset.columns, ' on ', q)
            idx = G_train.query(q).index.tolist() if q else G_train.index.tolist()
            if not len(idx):
                print('no samples found for ', q)
                continue
            y_train_idx = y_train[idx]
            if np.all(y_train_idx == 1.):
                y_train_idx[-1] = -1.
            elif np.all(y_train_idx == -1.):
                y_train_idx[-1] = 1.
            h = train_sklearn_linear_model(X_train[idx], G_train.iloc[idx], y_train_idx,
                                           method_name=curr_settings['method_name'],
                                           settings={**curr_settings, **{'labels': h_labels}},
                                           normalize_variables=False)
            candidate_models.append(h)
        decoupled.update_assignments(candidate_models)
        for i, ig in enumerate(data.group_encoder.groups):
            h_i = i + 1
            perf = decoupled.performance_map[h_i][ig][decoupled._assignment_sample]['perf']
            preds = decoupled.performance_map[h_i][ig][decoupled._assignment_sample]['preds']
            generic_perf = decoupled.performance_map[0][ig][decoupled._assignment_sample]['perf']
            generic_preds = decoupled.performance_map[0][ig][decoupled._assignment_sample]['preds']
            decoupled.assignments[ig] = {'model_index': h_i,
                                         'gain_over_generic': raw_gain(generic_perf, perf, decoupled.metric_name),
                                         'prob_change_over_generic': np.sum(np.not_equal(preds, generic_preds)) / len(
                                             preds)}
        for i, ig in enumerate(decoupled.reporting_groups):
            if MISSING_VALUE in ig:
                decoupled.assignments[ig] = {'model_index': 0,
                                             'gain_over_generic': 0.0,
                                             'prob_change_over_generic': 0.0}
        all_models['decoupled'] = decoupled

    model_names_order = settings['models']

    all_dfs = []
    auditor = PersonalizationAuditor(generic_model=generic_model, data=data, **settings)
    for model_type, model in all_models.items():
        for sample_type in settings['sample_types']:
            for metric_name in settings['metric_names']:
                if settings['table_type'] == 'data_use':
                    all_dfs.append(auditor.compute_data_use(model, model_type, metric_name, sample_type))
                elif settings['table_type'] == 'performance':
                    all_dfs.append(auditor.compute_envyfreeness_metrics(model, model_type, metric_name, sample_type,
                                                                        return_pvalues=True))
                    all_dfs.append(auditor.compute_rationality_metrics(model, model_type, metric_name, sample_type,
                                                                       return_pvalues=True))
                    all_dfs.append(auditor.compute_performance_metrics(model, model_type, metric_name, sample_type,
                                                                       return_type='dataframe', all_group_only=True))
                else:
                    raise NotImplementedError

    results = pd.concat(all_dfs, axis=0, ignore_index=True)

    out = dict(settings)
    out['models'] = all_models.update({'generic': auditor.generic_model})
    out['model_names_order'] = model_names_order
    out['results'] = results
    results_file = get_training_results_file(**settings)
    with open(results_file, 'wb') as outfile:
        dill.dump(out, outfile, protocol=dill.HIGHEST_PROTOCOL, recurse=True)
    print_log('saved results to {}'.format(results_file))
    return results_file


settings = {
    'data_name': 'apnea',
    'renegerate_all_exps': False,
    # 'models': ['onehot_impute', 'sequential', 'onehot', 'intersectional', 'flat', 'participatory_simple', 'decoupled'],
    'models': ['sequential'],
    'rebalancing_type': 'yg',
    'table_type': 'performance',
    #
    'fold_id': 'K05N01',
    'fold_num_validation': 4,
    'fold_num_test': 5,
    'method_name': 'logreg',
    #
    'random_seed': 2337,
    #
    'assignment_metric': 'auc',
    'assignment_sample': 'validation',  # training, validation
    'delta': 0.00001,
    'metric_names': METRIC_NAMES,
    'sample_types': ['validation'],
    'min_sample_size': 10,
}

ppid = os.getppid()  # Get parent process id
process_type = psutil.Process(ppid).name()  # ex pycharm, bash
print('process_type:', process_type)
if 'sh' in process_type:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str, help='Datasets to process, separate with spaces',
                        default=settings['data_name'])
    parser.add_argument('--renegerate-all-exps', action='store_true', help='Regenerate all experiments',
                        default=settings['renegerate_all_exps'])
    parser.add_argument('--models', nargs='+', help='Types of models to train',
                        default=settings['models'])
    parser.add_argument('--rebalancing-type', type=str, default=settings['rebalancing_type'],
                        help='Type of rebalancing to use')
    parser.add_argument('--table-type', type=str, default=settings['table_type'],
                        help='Table type: performance or data_use')

    parser.add_argument('--fold-id', type=str, default=settings['fold_id'], help='Fold ID')
    parser.add_argument('--fold-num-validation', type=int, default=settings['fold_num_validation'],
                        help='Fold Validation Number')
    parser.add_argument('--fold-num-test', type=int, default=settings['fold_num_test'], help='Fold Test Number')
    parser.add_argument('--method-name', type=str, default=settings['method_name'], help='Method Name (logreg)')

    parser.add_argument('--random-seed', type=int, default=settings['random_seed'], help='Random seed')

    parser.add_argument('--assignment-metric', type=str, default=settings['assignment_metric'],
                        help='Assignment metric for flat/seq')
    parser.add_argument('--assignment-sample', type=str, default=settings['assignment_sample'],
                        help='Assignment sample for flat/seq')

    parser.add_argument('--delta', type=float, default=settings['delta'],
                        help='Delta value for flat/sequential')
    parser.add_argument('--metric-names', nargs='+', default=settings['metric_names'],
                        help='Metrics to consider')
    parser.add_argument('--sample-types', nargs='+', default=settings['sample_types'],
                        help='Samples to consider')
    parser.add_argument('--min-sample-size', type=int, default=settings['min_sample_size'],
                        help='Min number of samples')
    args = parser.parse_args()
    print('args: ', args)
    settings.update(vars(args))

settings['seed'] = settings['random_seed']

train_models(settings)