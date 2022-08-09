"""
Helper classes to represent and manipulate datasets for a binary classification task
"""
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pathlib import Path
import warnings
from copy import copy
import dill
from dataclasses import dataclass, field
from typing import List
from imblearn.over_sampling import RandomOverSampler

from .groups import GroupAttributeSet, GroupAttributeEncoder
from .cross_validation import validate_cvindices


def convert_to_synth(data):
    def synth_y(row):
        if row['Sex'] == 'Female':
            return np.sign(row['x01'] - row['x02'])
        return np.sign(row['x01'] - row['x02']) if 'Age' == 'Old' else np.sign(row['x01'] - row['x02'] - row['x05'])

    new_y = np.asarray(data.df.apply(lambda row: synth_y(row), axis=1))
    new_y = np.where(new_y == 0, -1, new_y)
    data = BinaryClassificationDataset(X=data.X, y=new_y, group_df=data.group_df, cvindices=data._cvindices)
    return data


class BinaryClassificationDataset(object):
    """class to represent/manipulate a dataset for a binary classification task"""

    SAMPLE_TYPES = ('training', 'validation', 'test')

    def __init__(self, X, y, **kwargs):
        """

        :param X:
        :param y:
        :param kwargs:
        """
        self.group_df = pd.DataFrame(kwargs.get('group_df'))
        self.group_attributes = GroupAttributeSet(self.group_df)
        self.group_encoder = GroupAttributeEncoder(df=self.group_attributes.df,
                                                   encoding_type=kwargs.get('encoding_type', 'intersectional'))
        self.group_indices = self.group_encoder.to_indices(self.group_df)
        self.group_sizes = {g: {'n': sum(np.isin(self.group_indices, k)),
                                'n_neg': sum(y[np.isin(self.group_indices, k)] == -1),
                                'n_pos': sum(y[np.isin(self.group_indices, k)] == 1)} for k, g in
                            enumerate(self.group_encoder.groups)}

        # complete dataset
        self._full = BinaryClassificationSample(parent=self, X=X, y=y)

        # variable names
        self._names = BinaryClassificationVariableNames(parent=self, y=kwargs.get('y_name', 'y'),
                                                        X=kwargs.get('X_names',
                                                                     ['x%02d' % j for j in range(1, self.d + 1)]))

        # cvindices
        self._cvindices = kwargs.get('cvindices')
        self.n_pos = sum(y == 1)
        self.n_neg = sum(y == -1)

        # indicator to check if we have split into train, test, splits
        self.reset()

    def reset(self):
        """
        initialize data object to a state before CV
        :return:
        """
        self._fold_id = None
        self._fold_number_range = []
        self._fold_num_test = 0
        self._fold_num_range = 0
        self.training = self._full
        self.validation = self._full.filter(indices=np.zeros(self.n, dtype=np.bool_))
        self.test = self._full.filter(indices=np.zeros(self.n, dtype=np.bool_))
        assert self.__check_rep__()

    #### built-ins ####
    def __check_rep__(self):

        # check complete dataset
        assert self._full.__check_rep__()

        # check names
        assert self.names.__check_rep__()

        # check folds
        if self._cvindices is not None:
            validate_cvindices(self._cvindices)

        if self._fold_id is not None:
            assert self._cvindices is not None

        # check subsamples
        n_total = 0
        for sample_name in self.SAMPLE_TYPES:
            if hasattr(self, sample_name):
                sample = getattr(self, sample_name)
                assert sample.__check_rep__()
                n_total += sample.n

        assert self.n == n_total

        return True

    def __eq__(self, other):
        return (self._full == other._full) and \
               all(np.array_equal(self.cvindices[k], other.cvindices[k]) for k in self.cvindices.keys())

    def __len__(self):
        return self.n

    def __repr__(self):
        return f'ClassificationDataset<n={self.n}, d={self.d}>'

    def __copy__(self):

        cpy = BinaryClassificationDataset(
            X=self.X,
            y=self.y,
            X_names=self.names.X,
            y_name=self.names.y,
            group_df=self.group_attributes.df,
            cvindices=self.cvindices
        )

        return cpy

    #### io functions ####
    @staticmethod
    def read_csv(data_file, **kwargs):
        """
        loads raw data from CSV
        :param data_file: Path to the data_file
        :param helper_file: Path to the helper_file or None.
        :return:
        """
        # extract common file header from dataset file
        file_header = str(data_file).rsplit('_data.csv')[0]

        # convert file names into path objects with the correct extension
        files = {
            'data': '{}_data'.format(file_header),
            'helper': kwargs.get('helper_file', '{}_helper'.format(file_header)),
        }
        files = {k: Path(v).with_suffix('.csv') for k, v in files.items()}
        assert files['data'].is_file(), 'could not find dataset file: %s' % files['data']
        assert files['helper'].is_file(), 'could not find helper file: %s' % files['helper']

        # read helper file
        hf = pd.read_csv(files['helper'], sep=',')
        hf['is_variable'] = ~(hf['is_outcome'] | hf['is_group_attribute'])
        hf_headers = ['is_outcome', 'is_group_attribute', 'is_variable']
        assert all(hf[hf_headers].isin([0, 1]))
        assert sum(hf['is_outcome']) == 1, 'helper file should specify 1 outcome'
        assert sum(hf['is_variable']) >= 1, 'helper file should specify at least 1 variable'
        if sum(hf['is_group_attribute']) < 1:
            warnings.warn('dataset does not contain group attributes')

        # parse names
        names = {
            'y': hf.query('is_outcome')['header'][0],
            'G': hf.query('is_group_attribute')['header'].tolist(),
            'X': hf.query('is_variable')['header'].tolist(),
        }

        # specify expected data types
        dtypes = {names['y']: int}
        dtypes.update({n: str for n in names['G']})
        dtypes.update({n: float for n in names['X']})

        # read raw data from disk
        df = pd.read_csv(files['data'], sep=',', dtype=dtypes)
        assert set(df.columns.to_list()) == set(
            hf['header'].to_list()), 'helper file should contain metadata for every column in the data file'
        data = BinaryClassificationDataset(
            X=df[names['X']].values,
            y=df[names['y']].replace(0, -1).values,
            group_df=df[names['G']],
            X_names=names['X'],
            y_name=names['y'],
        )

        return data

    def save(self, file, overwrite=False, check_save=True):
        """
        saves object to disk
        :param file:
        :param overwrite:
        :param check_save:
        :return:
        """

        f = Path(file)
        if f.is_file() and overwrite is False:
            raise IOError('file %s already exists on disk' % f)

        # check data integrity
        assert self.__check_rep__()

        # save a copy to disk
        data = copy(self)
        data.reset()
        with open(f, 'wb') as outfile:
            dill.dump({'data': data}, outfile, protocol=dill.HIGHEST_PROTOCOL)

        if check_save:
            loaded_data = self.load(file=f)
            assert data == loaded_data

        return f

    @staticmethod
    def load(file):
        """
        loads processed data file from disk
        :param file: path of the processed data file
        :return: data and cvindices
        """
        f = Path(file)
        if not f.is_file():
            raise IOError('file: %s not found' % f)

        with open(f, 'rb') as infile:
            file_contents = dill.load(infile)
            assert 'data' in file_contents, 'could not find `data` variable in pickle file contents'
            assert file_contents['data'].__check_rep__(), 'loaded `data` has been corrupted'

        data = file_contents['data']
        return data

    #### variable names ####
    @property
    def names(self):
        """ pointer to names of X, y"""
        return self._names

    #### properties of the full dataset ####
    @property
    def n(self):
        """ number of examples in full dataset"""
        return self._full.n

    @property
    def d(self):
        """ number of features in full dataset"""
        return self._full.d

    @property
    def df(self):
        return self._full.df

    @property
    def X(self):
        """ feature matrix """
        return self._full.X

    @property
    def G(self):
        """DataFrame of group attributes"""
        return self._full.G

    @property
    def y(self):
        """ label vector"""
        return self._full.y

    @property
    def classes(self):
        return self._full.classes

    #### cross validation ####
    @property
    def cvindices(self):
        return self._cvindices

    @cvindices.setter
    def cvindices(self, cvindices):
        self._cvindices = validate_cvindices(cvindices)

    @property
    def fold_id(self):
        """string representing the indices of cross-validation folds
        K05N01 = 5-fold CV – 1st replicate
        K05N02 = 5-fold CV – 2nd replicate (in case you want to run 5-fold CV one more time)
        K10N01 = 10-fold CV – 1st replicate
        """
        return self._fold_id

    @fold_id.setter
    def fold_id(self, fold_id):
        assert self._cvindices is not None, 'cannot set fold_id on a BinaryClassificationDataset without cvindices'
        assert isinstance(fold_id, str), 'invalid fold_id'
        assert fold_id in self.cvindices, 'did not find fold_id in cvindices'
        self._fold_id = str(fold_id)
        self._fold_number_range = np.unique(self.folds).tolist()

    @property
    def folds(self):
        """integer array showing the fold number of each sample in the full dataset"""
        return self._cvindices.get(self._fold_id)

    @property
    def fold_number_range(self):
        """range of all possible training folds"""
        return self._fold_number_range

    @property
    def fold_num_validation(self):
        """integer from 1 to K representing the validation fold"""
        return self._fold_num_validation

    @property
    def fold_num_test(self):
        """integer from 1 to K representing the test fold"""
        return self._fold_num_test

    def split(self, fold_id, fold_num_validation=None, fold_num_test=None):
        """
        :param fold_id:
        :param fold_num_validation: fold to use as a validation set
        :param fold_num_test: fold to use as a hold-out test set
        :return:
        """

        if fold_id is not None:
            self.fold_id = fold_id
        else:
            assert self.fold_id is not None

        # parse fold numbers
        if fold_num_validation is not None and fold_num_test is not None:
            assert int(fold_num_test) != int(fold_num_validation)

        if fold_num_validation is not None:
            fold_num_validation = int(fold_num_validation)
            assert fold_num_validation in self._fold_number_range
            self._fold_num_validation = fold_num_validation

        if fold_num_test is not None:
            fold_num_test = int(fold_num_test)
            assert fold_num_test in self._fold_number_range
            self._fold_num_test = fold_num_test

        # update subsamples
        self.training = self._full.filter(
            indices=np.isin(self.folds, [self.fold_num_validation, self.fold_num_test], invert=True))
        self.validation = self._full.filter(indices=np.isin(self.folds, self.fold_num_validation))
        self.test = self._full.filter(indices=np.isin(self.folds, self.fold_num_test))
        return

    @property
    def summary(self):
        """
        :return: dataframe containing the number of positive and negative samples
        for each sample type (based on the current) split; the table shows
        the breakdown for all samples, as well as the samples for each
        intersectional group
        """

        group_names = [' & '.join(g).lower() for g in self.group_encoder.groups]
        group_counts = {
            'n': [getattr(self, s).group_sizes[g]['n'] for g in self.group_encoder.groups for s in self.SAMPLE_TYPES],
            'n_pos': [getattr(self, s).group_sizes[g]['n_pos'] for g in self.group_encoder.groups for s in
                      self.SAMPLE_TYPES],
            'group': [' & '.join(g).lower() for g in self.group_encoder.groups for _ in self.SAMPLE_TYPES],
            'sample_type': [s for _ in self.group_encoder.groups for s in self.SAMPLE_TYPES],
        }

        # Add total counts
        group_counts['n'] += [self.group_sizes[g]['n'] for g in self.group_encoder.groups]
        group_counts['n_pos'] += [self.group_sizes[g]['n_pos'] for g in self.group_encoder.groups]
        group_counts['group'] += [' & '.join(g).lower() for g in self.group_encoder.groups]
        group_counts['sample_type'] += ["total" for _ in self.group_encoder.groups]

        # Add group "all" counts
        group_counts['n'] += [getattr(self, s).n for s in self.SAMPLE_TYPES] + [self.n]
        group_counts['n_pos'] += [getattr(self, s).n_pos for s in self.SAMPLE_TYPES] + [self.n_pos]
        group_counts['group'] += ['all' for _ in self.SAMPLE_TYPES] + ["all"]
        group_counts['sample_type'] += [s for s in self.SAMPLE_TYPES] + ["total"]

        df = pd.DataFrame.from_records(group_counts)

        # create new columns
        df['n_neg'] = df['n'] - df['n_pos']
        df['p_pos'] = df['n_pos'] / df['n']

        # reorder columns
        df = df[['group', 'n', 'n_neg', 'n_pos', 'p_pos', 'sample_type']]

        # reorder rows
        df.sample_type = df.sample_type.astype(
            CategoricalDtype(['training', 'validation', 'test', 'total'], ordered=True))
        df.group = df.group.astype(CategoricalDtype(group_names + ['all'], ordered=True))
        df = df.sort_values(by=['group', 'sample_type'])
        return df


@dataclass
class BinaryClassificationSample:
    """class to store and manipulate a subsample of points in a survival dataset"""

    parent: BinaryClassificationDataset
    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray = None

    def __post_init__(self):

        self.classes = (-1, 1)
        self.X = np.atleast_2d(np.array(self.X, float))
        self.n = self.X.shape[0]
        self.n_pos = sum(self.y == 1)
        self.n_neg = sum(self.y == -1)
        self.d = self.X.shape[1]
        self.group_encoder = self.parent.group_encoder

        if self.indices is None:
            self.indices = np.ones(self.n, dtype=np.bool_)
        else:
            self.indices = self.indices.flatten().astype(np.bool_)

        self.update_classes(self.classes)
        assert isinstance(self.G, pd.DataFrame)
        assert self.__check_rep__()

        self.group_indices = self.group_encoder.to_indices(self.G)
        self.group_sizes = {g: {'n': sum(np.isin(self.group_indices, k)),
                                'n_neg': sum(self.y[np.isin(self.group_indices, k)] == -1),
                                'n_pos': sum(self.y[np.isin(self.group_indices, k)] == 1)} for k, g in
                            enumerate(self.group_encoder.groups)}

    def __len__(self):
        return self.n

    def __eq__(self, other):
        chk = isinstance(other, BinaryClassificationSample) and np.array_equal(self.y, other.y) and np.array_equal(
            self.X, other.X)
        return chk

    def __check_rep__(self):
        """returns True is object satisfies representation invariants"""
        assert isinstance(self.X, np.ndarray)
        assert isinstance(self.y, np.ndarray)
        assert self.n == len(self.y)
        assert np.sum(self.indices) == self.n
        assert np.isfinite(self.X).all()
        assert np.isin(self.y, self.classes).all(), 'y values must be stored as {}'.format(self.classes)
        return True

    @property
    def G(self):
        """matrix of group attributes"""
        return self.parent.group_attributes.df[self.indices]

    def update_classes(self, values):
        assert len(values) == 2
        assert values[0] < values[1]
        assert isinstance(values, (np.ndarray, list, tuple))
        self.classes = tuple(np.array(values, dtype=int))

        # change y encoding using new classes
        if self.n > 0:
            y = np.array(self.y, dtype=float).flatten()
            neg_idx = np.equal(y, self.classes[0])
            y[neg_idx] = self.classes[0]
            y[~neg_idx] = self.classes[1]
            self.y = y

    @property
    def df(self):
        """
        pandas data.frame containing y, G, X for this sample
        """
        df = pd.DataFrame(self.X, columns=self.parent.names.X)
        df = pd.concat([self.G, df], axis=1).reset_index(drop=True)
        df.insert(column=self.parent.names.y, value=self.y, loc=0).reset_index(drop=True)
        return df

    #### methods #####
    def filter(self, indices):
        """filters samples based on indices"""
        assert isinstance(indices, np.ndarray)
        assert indices.ndim == 1 and indices.shape[0] == self.n
        assert np.isin(indices, (0, 1)).all()
        return BinaryClassificationSample(parent=self.parent, X=self.X[indices], y=self.y[indices], indices=indices)


@dataclass
class BinaryClassificationVariableNames:
    """class to represent the names of features, group attributes, and the label in a classification task"""
    parent: BinaryClassificationDataset
    X: List[str] = field(repr=True)
    y: str = field(repr=True, default='y')

    def __post_init__(self):
        assert self.__check_rep__()

    @property
    def G(self):
        return self.parent.group_attributes.names

    @staticmethod
    def check_name_str(s):
        """check variable name"""
        return isinstance(s, str) and len(s.strip()) > 0

    def __check_rep__(self):
        """check if this object satisfies representation invariants"""

        assert isinstance(self.X, list) and all([self.check_name_str(n) for n in self.X]), 'X must be a list of strings'
        assert len(self.X) == len(set(self.X)), 'X must be a list of unique strings'

        assert isinstance(self.G, list) and all([self.check_name_str(n) for n in self.G]), 'G must be a list of strings'
        assert len(self.G) == len(set(self.G)), 'G must be a list of unique strings'

        assert self.check_name_str(self.y), 'y must be at least 1 character long'
        return True


def oversample_by_label(data, **kwargs):
    """
    oversample dataset to equalize number of positive and negative labels in each group
    :param data:
    :param kwargs:
    :return:
    """

    group_df = data.group_attributes.df
    group_indices = data.group_encoder.to_indices(group_df)
    ros = RandomOverSampler(**kwargs)

    # generate resampled data
    Xr, yr, Gr = [], [], []
    for k, g in enumerate(data.group_encoder.groups):
        idx = np.isin(group_indices, k)
        Xg, yg = ros.fit_resample(data.X[idx, :], data.y[idx])
        Xr.append(Xg)
        yr.append(yg)
        Gr.append(np.tile(g, (len(yg), 1)))

    # concatenate
    Xr = np.vstack(Xr)
    yr = np.concatenate(yr)
    Gr = pd.DataFrame(np.vstack(Gr), columns=data.G.columns)

    # return new dataset object
    return BinaryClassificationDataset(X=Xr, y=yr, group_df=Gr)


def oversample_by_group_and_label(data, **kwargs):
    """
    oversample dataset to equalize number of positive and negative labels in each group and the size of each group
    :param data:
    :param kwargs:
    :return:
    """
    m = len(data.group_attributes)
    group_df = data.group_attributes.df
    group_indices = data.group_encoder.to_indices(group_df)

    # generate ids for each unique combination (G, y)
    group_values_with_label = np.concatenate((group_indices[:, None], data.y[:, None]), axis=1)
    _, profile_idx = np.unique(group_values_with_label, axis=0, return_inverse=True)

    # oversample groups and labels
    ros = RandomOverSampler(**kwargs)
    D = np.concatenate((data.G, data.X, data.y[:, None]), axis=1)
    D, T = ros.fit_resample(D, profile_idx)
    _, profile_counts = np.unique(T, axis=0, return_counts=True)
    assert np.all(profile_counts == profile_counts[0])

    # split
    X_res = D[:, m:(m + data.d)]
    y_res = D[:, -1]
    G_res = pd.DataFrame(data=D[:, :m], columns=data.G.columns)

    return BinaryClassificationDataset(X=X_res, y=y_res, group_df=G_res)
