"""
This script creates processed datasets from raw data files in

`psc/data/[data_name]/`

The raw data files required for each dataset are:
- [data_name]_data.csv which contains a table of [y, X] values without missing data
- [data_name]_helper.csv which contains metadata for each column in [data_name]_data.csv
"""
import os
import psutil

from psc.paths import data_dir, get_processed_data_file
from psc.data import BinaryClassificationDataset, oversample_by_label, oversample_by_group_and_label
from psc.cross_validation import generate_cvindices

settings = {
    'data_names': ['apnea'],
    'random_seed': 2338,
    'rebalancing_types': [
        'none',  # no rebalancing
        'yperg',  # equalize positive and negative samples for each group
        'yg'  # equalize positive and negative samples for each group, then equalize number of samples per group
    ]}

ppid = os.getppid()  # Get parent process id
process_type = psutil.Process(ppid).name()  # ex pycharm, bash
print('process_type:', process_type)
if 'sh' in process_type:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-names', nargs='+', help='Datasets to process, separate with spaces',
                        default=settings['data_names'])
    parser.add_argument('--random-seed', type=int, default=settings['random_seed'], help='Random seed')
    parser.add_argument('--rebalancing-types', nargs='+', help='Rebalancing Types', default=settings['rebalancing_types'])
    args = parser.parse_args()
    print('args: ', args)
    settings.update(vars(args))

for data_name in settings['data_names']:
    for rebalancing_type in settings['rebalancing_types']:
        data_file_processed = get_processed_data_file(data_name, rebalancing_type)
        print(f'processing {data_name}')
        data_file_raw = data_dir / data_name / '{}_data.csv'.format(data_name)

        # create a dataset object by reading a CSV from disk
        data = BinaryClassificationDataset.read_csv(data_file=data_file_raw)
        if rebalancing_type == 'yperg':
            data = oversample_by_label(data)
        elif rebalancing_type == 'yg':
            data = oversample_by_group_and_label(data)

        # generate indices for stratified cross-validation
        data.cvindices = generate_cvindices(strata=data.y,
                                            total_folds_for_cv=[1, 3, 4, 5],
                                            total_folds_for_inner_cv=[],
                                            replicates=3,
                                            seed=settings['random_seed'])

        # save processed file
        data.save(file=data_file_processed, overwrite=True, check_save=True)
        print(f"Processed data saved to {data_file_processed}")
