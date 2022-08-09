"""
Classes to represent, manipulate and encode group attributes
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pandas.api.types import CategoricalDtype
import itertools

from sklearn.preprocessing import PolynomialFeatures

MISSING_VALUE = '∅'


def is_subgroup(g1, g2):
    return all([(g1a == g2a or g2a == MISSING_VALUE) for g1a, g2a in zip(g1, g2)])


@dataclass
class GroupAttribute:
    """
    helper class to represent and manipulate a group attribute G
    -----
    series: pd.Series containing group attributes
    indexer: dictionary of the form: {G[k]: k} where G[k] is the group label and k is the integer index
    -----
    Representation

    G is a categorical attribute with the form

        G = {G[1], G[2], ..., G[m]} where m ≥ 2

    For example:

        Sex = {Male, Female}
        AgeGroup = {Age_lt_30, Age_30_to_50, Age_gt_50}

    We refer to 'G' as a 'group attribute', and a 'G[k]' as a 'label'.
    """
    series: pd.Series = field(repr=False, init=True)
    labels: list = None

    def __post_init__(self):
        if self.labels is None:
            self.labels = self.series.unique().tolist()
            self.labels.sort()

        # check that there exist at least two labels
        assert set(self.labels).issuperset(self.series), 'labels[{}] is missing labels observed in df[{}]'.format(
            self.name, self.name)
        assert len(self.labels) >= 2, 'group attribute {} has fewer than distinct 2 labels'.format(self.name)

        # re-encode series to categorical using labels
        self.series = self.series.astype(dtype=CategoricalDtype(categories=self.labels))

        # convert values into indices
        self.dummy_df = pd.get_dummies(data=self.series, prefix=self.name, prefix_sep=':')

    @property
    def name(self):
        """name of the group attribute"""
        return self.series.name

    @property
    def dummy_values(self):
        """matrix of indicator variables or group membership [1[G=G[0], ... , 1[G=G[m]]"""
        return self.dummy_df.values

    @property
    def dummy_names(self):
        """names of indicator variables for group membership"""
        return self.dummy_df.columns.to_list()

    @property
    def tally(self):
        return self.series.value_counts(sort=False)[self.labels]

    def __repr__(self):
        tally_str = ' ,'.join(f'{i[0]}={i[1]}' for i in self.tally.iteritems())
        return f'<GroupAttribute({self.name}={self.labels}, total={self.series.shape[0]}, {tally_str}>'


class GroupAttributeEncoder(object):
    """
    helper class to create a encoder
    encoders are designed to implement a `to_dummies` method,
    this method transforms a data frame of categorical group attributes
    into a numeric array that then be used to train classifiers
    """

    ENCODING_TYPES = ('onehot', 'intersectional', 'ordinal')

    def __init__(self, df, encoding_type, labels=None, **kwargs):
        """

        :param df:
        :param encoding_type:
        :return:
        """
        assert isinstance(df, pd.DataFrame)
        names = df.columns.tolist()
        assert not any([(':' in n) for n in names])
        assert not any([(';' in n) for n in names])
        self._names = names
        self._encoding_type = encoding_type
        assert self._encoding_type in GroupAttributeEncoder.ENCODING_TYPES

        # create a dictionary of indices for each column in df
        if labels is None:
            attributes = [GroupAttribute(series=df[name]) for name in df]
        else:
            assert isinstance(labels, dict)
            assert set(labels.keys()).issubset(df.columns)
            attributes = [GroupAttribute(series=df[name], labels=labels[name]) for name in df]
        # specify labels and dtypes for each group attribute
        self._labels = {g.name: g.labels for g in attributes}
        self._dtypes = {name: pd.CategoricalDtype(categories=labels) for name, labels in self._labels.items()}

        # specify labels for intersectional groups
        self._intersectional_labels = list(itertools.product(*self.labels.values()))
        self.include_interactions = kwargs.get('include_interactions', False)

        # setup dummy names
        if self._encoding_type == 'onehot':
            dummy_names = np.hstack([g.dummy_names for g in attributes]).tolist()
        elif self._encoding_type == 'intersectional':
            dummy_names = itertools.product(*[g.dummy_names for g in attributes])
            dummy_names = [' & '.join(n) for n in dummy_names]
            self.indexer = {label: i for i, label in enumerate(self._intersectional_labels)}
        elif self._encoding_type == 'ordinal':
            if self.include_interactions:
                mapper = {v: i for ls in self.labels.values() for i, v in enumerate(sorted(ls))}
                df = df[self.names].replace(mapper)
                poly = PolynomialFeatures(interaction_only=True, include_bias=False).fit(df)
                dummy_names = poly.get_feature_names_out()
            else:
                dummy_names = self._names

        self._dummy_names = dummy_names
        self.mode = df.mode()

    @property
    def names(self):
        """names of the group attributes"""
        return self._names

    @property
    def labels(self):
        """dictionary of the form {group_attribute: group_attribute_labels}"""
        return self._labels

    @property
    def groups(self):
        """list of tuples containing the labels of intersectional groups"""
        return self._intersectional_labels

    @property
    def encoding_type(self):
        """string representing how the group attributes are currently encoded"""
        return self._encoding_type

    @property
    def dummy_names(self):
        """list containing the names of the dummy variables"""
        return self._dummy_names

    def to_dummies(self, df, return_names=False):
        """
        :param df: pandas.DataFrame containing raw group attributes
        :param return_names: set as True to return dummy names
        :return:
        """
        assert isinstance(df, pd.DataFrame), 'type is ' + str(type(df)) + ' not pd.DataFrame'
        # reorder columns
        if self.encoding_type == 'onehot':
            df = pd.get_dummies(data=df[self.names].astype(dtype=self._dtypes), prefix=self.names, prefix_sep=':')
            values = df.values.astype(np.int_)
        elif self.encoding_type == 'intersectional':
            indices = np.array(list(map(lambda x: self.indexer.get(tuple(x)), df.values)))
            values = np.array([np.isin(indices, k) for k in self.indexer.values()]).transpose().astype(np.int_)
        elif self.encoding_type == 'ordinal':
            mapper = {v: i for ls in self.labels.values() for i, v in enumerate(sorted(ls))}
            df = df[self.names].replace(mapper)
            if self.include_interactions:
                poly = PolynomialFeatures(interaction_only=True, include_bias=False).fit(df)
                values = poly.transform(df).astype(np.int_)
            else:
                values = df.values.astype(np.int_)
        if return_names:
            return values, self.dummy_names
        else:
            return values

    def to_indices(self, df, group_to_index=None):
        """
        converts DataFrame of group attributes into an array of integer indices
        indices represent the intersectional group of each row
        :param df: pandas.DataFrame containing raw group attributes
        :return: array of integers from 0 to len(self.groups)
        """
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns.tolist()) == set(self.names)
        if group_to_index is None:
            group_to_index = {g: i for i, g in enumerate(self._intersectional_labels)}
        indices = df[self.names].apply(func=lambda x: group_to_index.get(tuple(x)), axis=1).values
        return indices


@dataclass
class GroupAttributeSet:
    """
    helper class to represent a set of GroupAttributes
    this class also provides some of the functionality in GroupAttributeEncoder
    """
    df: pd.DataFrame = field(repr=False, init=True)
    attributes: dict = field(repr=True, init=False)
    ENCODING_TYPES = ('onehot', 'intersectional', 'ordinal')

    def __post_init__(self):

        self._encoding_type = 'onehot'

        # initialize attribute objects
        self.attributes = [GroupAttribute(series=self.df[name]) for name in self.df]

        # recreate df from attributes
        self.df = pd.concat([g.series for g in self.attributes], axis=1)

        # compute all labels and intersectional dummy names
        intersectional_labels = itertools.product(*[g.labels for g in self.attributes])
        intersectional_dummy_names = itertools.product(*[g.dummy_names for g in self.attributes])
        self.indexer = {label: i for i, label in enumerate(intersectional_labels)}

        # setup d names / values
        self._dummy_names = {
            'onehot': np.hstack([g.dummy_names for g in self.attributes]).tolist(),
            'intersectional': [' & '.join(n) for n in list(intersectional_dummy_names)],
            'ordinal': self.names,
        }

        self._dummy_values = {
            'onehot': np.hstack([g.dummy_values for g in self.attributes]),
            'intersectional': np.array(
                [np.isin(list(map(lambda x: self.indexer.get(tuple(x)), self.values)), k) for k in
                 self.indexer.values()]).transpose(),
            'ordinal': self.names
        }

    def __len__(self):
        return len(self.attributes)

    @property
    def names(self):
        """names of the group attributes"""
        return self.df.columns.tolist()

    @property
    def values(self):
        """raw values of the group attributes"""
        return self.df.values.astype(np.str_)

    @property
    def labels(self):
        """dictionary {group_attribute: group_attribute_labels}"""
        return {g.name: g.labels for g in self.attributes}

    @property
    def encoding_type(self):
        """a string representing how the group attributes are currently encoded"""
        return self._encoding_type

    @encoding_type.setter
    def encoding_type(self, value):
        """change the encoding type: must be 'onehot' or 'intersectional' or 'ordinal'"""
        assert value in GroupAttributeSet.ENCODING_TYPES
        self._encoding_type = str(value)

    @property
    def dummy_names(self):
        """list containing the names of the dummy variables"""
        return self._dummy_names.get(self._encoding_type)

    @property
    def dummy_values(self):
        """numpy array containing dummy variables"""
        return self._dummy_values.get(self._encoding_type)

    #### methods ####

    def to_dummies(self, encoding_type='raw', selection=None, return_names=False):
        """
        :param encoding_type:
        :param selection:
        :param return_names:
        :return:
        """
        assert encoding_type == 'raw' or encoding_type in GroupAttributeSet.ENCODING_TYPES

        # quick return if we are returning raw/decoupled/intersectional
        if selection is None:

            if encoding_type == 'raw':
                values = self.values.astype(np.str_)
                names = self.names
            else:
                values = self._dummy_values.get(encoding_type).astype(np.int_)
                names = self._dummy_names.get(encoding_type)

        else:

            # otherwise we need to recompute values
            assert isinstance(selection, list)
            assert set(selection).issubset(self.names)
            assert len(selection) == len(set(selection))

            if encoding_type == 'raw':
                values = self.df[selection].values.astype(np.str_)
                names = [str(g) for g in selection]

            elif encoding_type == 'onehot' or encoding_type == 'ordinal':
                selected_attributes = [g for g in self.attributes if g.name in selection]
                values = np.hstack([g.dummy_values for g in selected_attributes]).astype(np.int_)
                names = [g.dummy_names for g in selected_attributes]

            elif encoding_type == 'intersectional':
                selected_attributes = [g for g in self.attributes if g.name in selection]
                intersectional_labels = itertools.product(*[g.labels for g in selected_attributes])
                indexer = {label: i for i, label in enumerate(intersectional_labels)}
                indices = np.array(list(map(lambda x: indexer.get(tuple(x)), self.df[selection].values)))
                values = np.array([np.isin(indices, k) for k in indexer.values()]).transpose().astype(np.int_)
                names = [g.dummy_names for g in selected_attributes]
        if return_names:
            return values, names

        return values
