import pandas as pd
import dill
from datetime import datetime

from psc.paths import get_training_results_file, results_dir
from psc.utils import METRIC_NAMES
from scripts.train_and_benchmark import train_models

settings = {
    'data_names': ['apnea'],
    'renegerate_all_exps': False,
    # 'models': ['sequential', 'flat', 'participatory_simple', 'onehot_impute', 'intersectional', 'onehot'],
    'models': ['sequential'],
    'rebalancing_type': 'yg',
    #
    'fold_id': 'K05N01',
    'fold_num_validation': 4,
    'fold_num_test': 5,
    'method_name': 'logreg',
    #
    'seed': 2337,
    #
    'assignment_metric': 'auc',
    'assignment_sample': 'validation',  # training, validation
    'delta': 0.00001,
    'metric_names': METRIC_NAMES,
    'sample_types': ['validation'],
    'min_sample_size': 10,
}

from tqdm import tqdm

data_use_all_results = []
performance_all_results = []
for data_name in tqdm(settings['data_names']):
    for model in settings['models']:
        curr_settings = settings.copy()
        curr_settings.update({'models': [model], 'data_name': data_name})
        f_data_use = get_training_results_file(**{**curr_settings, 'table_type': 'data_use'})
        f_performance = get_training_results_file(**{**curr_settings, 'table_type': 'performance'})
        if not f_data_use.exists() or settings['renegerate_all_exps']:
            print('GENERATING ', f_data_use)
            train_models({**curr_settings, 'table_type': 'data_use'})
        if settings['renegerate_all_exps'] or not f_performance.exists():
            print('GENERATING ', f_performance)
            train_models({**curr_settings, 'table_type': 'performance'})
        with open(f_data_use, 'rb') as infile:
            data_use_results = dill.load(infile)
        with open(f_performance, 'rb') as infile:
            performance_results = dill.load(infile)
        # todo: append settings FROM results files
        if 'assignment_metric' not in data_use_results['results'].columns:
            data_use_results['assignment_metric'] = settings['assignment_metric']
        if 'assignment_metric' not in performance_results['results'].columns:
            performance_results['assignment_metric'] = settings['assignment_metric']
        if 'assignment_sample' not in data_use_results['results'].columns:
            data_use_results['assignment_sample'] = settings['assignment_sample']
        if 'assignment_sample' not in performance_results['results'].columns:
            performance_results['assignment_sample'] = settings['assignment_sample']
        data_use_all_results.append(data_use_results['results'])
        performance_all_results.append(performance_results['results'])

### aggregate results
data_use_all_results_df = pd.concat(data_use_all_results, axis=0, ignore_index=True)
print(
    f"data use aggregate results: {data_use_all_results_df.shape[0]} rows by {data_use_all_results_df.shape[1]} columns")

perf_all_results_df = pd.concat(performance_all_results, axis=0, ignore_index=True)
print(f"perf aggregate results: {perf_all_results_df.shape[0]} rows by {perf_all_results_df.shape[1]} columns")

### aggregated results
now = datetime.now()
time_stamp = now.strftime("%m_%d_%H%M")

# save performance statistics as CSV
performance_file = results_dir / f'performance_table_check_{settings["assignment_metric"]}_{settings["rebalancing_type"]}_{time_stamp}.csv'
performance_df = perf_all_results_df.drop('model', axis=1)
performance_df.to_csv(performance_file, index=False)
print('performance results saved to: ', performance_file)

# save data use CSV
data_use_file = results_dir / f'data_use_check_{settings["assignment_metric"]}_{settings["rebalancing_type"]}_{time_stamp}.csv'
data_use_df = data_use_all_results_df.drop('model', axis=1)
data_use_df.to_csv(data_use_file, index=False)
print('data use results saved to : ', data_use_file)

print('done')
