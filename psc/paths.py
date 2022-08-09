"""
This file defines paths for key directories and files. Contents include:
1. Directory Names: Path objects that specify the directories where we store code, data, results, etc.
2. File Name Generators: functions used to programatically name processed datasets, results, graphs etc.
"""

from pathlib import Path

# Directories

# path to the GitHub repository
repo_dir = Path(__file__).resolve().parent.parent

# path to the Python package
pkg_dir = repo_dir / "psc/"
# pkg_dir = Path("psc")

# directory where we store datasets
data_dir = repo_dir / "data/"

# directory where we store results
results_dir = repo_dir / "results/"

# directory where we store reports
reports_dir = repo_dir / "reports/"

# directory of reporting package
reporting_dir = repo_dir / 'reporting/'

# directory where we store templates
templates_dir = repo_dir / 'templates/'

# create local directories if they do not exist
results_dir.mkdir(exist_ok=True)
reports_dir.mkdir(exist_ok=True)


# Naming Functions
def get_processed_data_file(data_name, rebalancing_type, **kwargs):
    """
    :param data_name: string containing name of the dataset
    :param data_name: string containing rebalancing_type code
    :param kwargs: used to catch other args when unpacking dictionaries
                   this allows us to call this function as get_results_file_name(**settings)
    :return:
    """
    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(rebalancing_type, str) and len(rebalancing_type) > 0
    f = data_dir / '{}_{}_processed.pickle'.format(data_name, rebalancing_type)
    return f


def get_training_results_file(data_name, rebalancing_type, method_name, fold_id, assignment_metric,
                              assignment_sample, **kwargs):
    """
    returns file name for pickle files used to store the results of a training job (e.g., in `train_classifier`)

    :param data_name: string containing name of the dataset
    :param method_name: string containing name of the classification method
    :param encoding_type: string how group attributes were encoded
    :param fold_id: string specifying fold_id used for training
    :param assignment_metric: string metric for assigning models in participatory systems
    :param assignment_sample: string sample for assigning models in participatory system
    :param kwargs: used to catch other args when unpacking dictionaies
                   this allows us to call this function as get_results_file_name(**settings)

    :return: Path of results object
    """

    assert isinstance(data_name, str) and len(data_name) > 0
    assert isinstance(method_name, str) and len(method_name) > 0
    assert isinstance(rebalancing_type, str) and len(rebalancing_type) > 0
    assert isinstance(fold_id, str) and len(fold_id) > 0
    assert isinstance(assignment_metric, str) and len(assignment_metric) > 0
    assert isinstance(assignment_sample, str) and len(assignment_sample) > 0
    suffix = ""
    if kwargs.get('baseline', False):
        suffix += "_baseline"
    else:
        suffix += f"_t{kwargs['selected_tree_index']}" if 'selected_tree_index' in kwargs else ""
        suffix += "_"
    if 'models' in kwargs:
        suffix += "_".join([m for m in kwargs['models']])

    desc = '' if 'desc' not in kwargs else '_' + kwargs['desc']

    f = '{}_{}_{}_{}_{}_{}{}{}.results'.format(data_name, rebalancing_type, method_name,
                                               fold_id, assignment_metric,
                                               assignment_sample, suffix, desc)
    if 'table_type' in kwargs:
        f = f'{kwargs["table_type"]}_{f}'
    f = results_dir / f
    return f
