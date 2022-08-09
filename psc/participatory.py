import numpy as np
import os

from anytree import Node, RenderTree, PreOrderIter, Walker
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont

from psc import metrics
from psc.data import BinaryClassificationDataset
from psc.classification import ClassificationModel
from psc.groups import MISSING_VALUE, is_subgroup
from psc.utils import METRIC_NAMES, product, clean_latex_str, metric_latex, \
    group_name_latex, raw_gain

# NOTE: these constants should be changed both here and in corresponding .Rmd files
HUE = 30
H_COLOR = 'cyan'
H0_COLOR = 'lightgray'


class PersonalizedSystem(object):
    """
    Interface for personalized systems
    """

    def __init__(self, data, generic_model,
                 metric_name='auc',
                 assignment_sample='training',
                 delta=0.0, **kwargs):
        """
        :param data: BinaryClassificationDataset
        :param generic_model: ClassificationModel -- model that doesn't require group information
        :param metric_name: e.g. auc, ece, log_loss, error, etc
        :param assignment_sample: e.g. train, validation, test, ''
        """
        assert isinstance(data, BinaryClassificationDataset)
        assert isinstance(generic_model, ClassificationModel)
        assert assignment_sample in list(BinaryClassificationDataset.SAMPLE_TYPES) + ['']
        assert metric_name in METRIC_NAMES

        # data initialization
        self.data = data
        self.attribute_names = data.group_attributes.names
        self.reporting_options = {k: v + [MISSING_VALUE] for k, v in data.group_attributes.labels.items()}
        self.reporting_groups = list(product(*self.reporting_options.values()))
        self._assignment_sample = assignment_sample
        self.dnr_all_group = tuple([MISSING_VALUE] * len(self.reporting_options))

        # initialize performance metric
        self.metric_name = metric_name

        # initialize update rule
        self._delta = delta

        # model initialization
        self.generic_model = generic_model
        # assignment initialization
        self.assignments = {ig: {'model_index': 0, 'gain_over_generic': 0.0, 'prob_change_over_generic': 0.0} for ig in
                            self.reporting_groups}
        self.name_index_map = {name: i for i, name in enumerate(self.attribute_names)}
        self.models = [self.generic_model]
        self._ig_filtered_map = None
        self._performance_map = None

    @property
    def ig_filtered_map(self):
        if self._ig_filtered_map is None:
            self._set_ig_filtered_map()
        assert self._ig_filtered_map is not None
        return self._ig_filtered_map

    @property
    def full_reporting_groups(self):
        return list(set([self.ig_filtered_map[rg] for rg in self.reporting_groups if not rg.count(MISSING_VALUE)]))

    @property
    def required_group_attributes(self):
        raise NotImplementedError

    @property
    def performance_map(self):
        if self._performance_map is None:
            self._set_performance_map()
        assert self._performance_map is not None
        return self._performance_map

    def _set_ig_filtered_map(self):
        """
        Create map from reporting group to filtered reporting group,
        such that the filtered reporting group contains only data
        used by the model
        :return:
        """
        raise NotImplementedError

    def _set_performance_map(self):
        """
        Create map of models to performance statistics on all intersectional groups
        used by the model
        :return:
        """
        self._performance_map = defaultdict(dict)
        for i, h in enumerate(self.models):
            for ig in self.reporting_groups:
                self.performance_map[i][ig] = defaultdict(dict)
                for sample_type in ['training', 'validation', 'test']:
                    perf, sample_size, preds = self.compute_performance(h, ig, sample_type=sample_type)
                    self._performance_map[i][ig][sample_type] = {'perf': perf, 'sample_size': sample_size,
                                                                 'preds': preds}

    def check_fair_use(self, h_i, ig):
        for subgroup in self._get_subgroups(ig):
            h0_train_perf = self.performance_map[0][subgroup]['training']['perf']
            h0_perf = self.performance_map[0][subgroup][self._assignment_sample]['perf']
            h_i_train_perf = self.performance_map[h_i][subgroup]['training']['perf']
            h_i_perf = self.performance_map[h_i][subgroup][self._assignment_sample]['perf']
            if np.isnan(h_i_train_perf) or np.isnan(h_i_perf):
                return False
            gain_train = raw_gain(h0_train_perf, h_i_train_perf, self.metric_name)
            gain_sample_metric = raw_gain(h0_perf, h_i_perf, self.metric_name)
            if gain_train <= 0 or gain_sample_metric <= 0:
                return False
        return True

    def update_rule(self, m1_i, m2_i, ig):
        if m1_i == m2_i:
            return False
        m1_perf = self.performance_map[m1_i][ig][self._assignment_sample]['perf']
        m2_perf = self.performance_map[m2_i][ig][self._assignment_sample]['perf']
        return raw_gain(m1_perf, m2_perf, self.metric_name) > self._delta

    def compute_performance(self, clf, intersectional_group, sample_type=None, metric_name=None):
        """
        Compute the performance of classifer h on reporting group r
        If there are no samples of intersectional_group or additional attributes are required, return np.nan
        :param clf: ClassificationModel
        :param intersectional_group: tuple e.g. (Male, Old)
        :param sample_type: training, validation, test
        :param metric_name: ece, log_loss, auc, error
        :return: [performance, sample_size, predictions]
        """
        sample_type = sample_type if sample_type is not None else self._assignment_sample
        # self._assignment_sample can be None -- default to full dataset
        data_sample = getattr(self.data, sample_type) if sample_type is not None else self.data
        G = data_sample.G.reset_index(drop=True)
        X = data_sample.X
        y = data_sample.y
        metric_name = metric_name if metric_name is not None else self.metric_name
        metric_handle = getattr(metrics, 'compute_{}'.format(metric_name))
        prediction_handle = getattr(clf, metrics.predict_handles_map[metric_name] + '_for')
        y_pred, idx = prediction_handle(X, G, intersectional_group, use_missing_val=False)
        pred_vals, _ = clf.predict_for(X, G, intersectional_group, use_missing_val=False)
        res = [float('nan'), len(idx), pred_vals]
        if len(idx):
            res[0] = metric_handle(y[idx], y_pred)
        return res

    def _reset_vars(self):
        """reset variables before updating assignments"""
        self._ig_filtered_map = None

    def update_assignments(self, candidate_models, force=False):
        """
        updates assignments given candidate models
        :param candidate_models: ClassificationModel list
        :param force: force override candidate models
        :return: None
        """
        self._reset_vars()
        assert isinstance(candidate_models, list)
        assert len(self.models) == 1 or force, "ERROR models already set, use force=True to overwrite"
        assert all(isinstance(h, ClassificationModel) for h in candidate_models)
        self.models = [self.generic_model] + candidate_models
        self._set_performance_map()
        self.assignments = {ig: {'model_index': 0, 'gain_over_generic': 0.0, 'prob_change_over_generic': 0.0} for ig in
                            self.reporting_groups}
        for ig in self.reporting_groups:
            valid_inds = list(filter(lambda h_i: self.check_fair_use(h_i, ig), list(range(len(self.models)))))
            if not len(valid_inds):
                continue
            gen_perf = self.performance_map[0][ig][self._assignment_sample]['perf']
            ind = max(valid_inds,
                      key=lambda i: raw_gain(gen_perf,
                                             self.performance_map[i][ig][self._assignment_sample]['perf'],
                                             self.metric_name))
            # h_ind is not sufficiently better than currently assigned model (h_0)
            if not self.update_rule(0, ind, ig):
                continue
            perf = self.performance_map[ind][ig][self._assignment_sample]['perf']
            preds = self.performance_map[ind][ig][self._assignment_sample]['preds']
            generic_perf = self.performance_map[0][ig][self._assignment_sample]['perf']
            generic_preds = self.performance_map[0][ig][self._assignment_sample]['preds']
            # prob_change_over_generic may be 0 even when gain_over_generic != 0 (e.g., log_loss depends on proba values
            # and may change even when predictions do not)
            self.assignments[ig] = {'model_index': ind,
                                    'gain_over_generic': raw_gain(generic_perf, perf, self.metric_name),
                                    'prob_change_over_generic': np.sum(np.not_equal(preds, generic_preds)) / len(preds)}

    def _get_supergroups(self, ig, candidate_igs=None):
        c_igs = candidate_igs if candidate_igs is not None else self.reporting_groups
        supergroups = list(filter(lambda c_ig: is_subgroup(ig, c_ig), c_igs))
        # sort specific to general
        supergroups.sort(key=lambda ig: ig.count(MISSING_VALUE))
        return supergroups

    def _get_subgroups(self, ig, candidate_igs=None):
        c_igs = candidate_igs if candidate_igs is not None else self.reporting_groups
        subgroups = list(filter(lambda c_ig: is_subgroup(c_ig, ig), c_igs))
        # sort specific to general
        subgroups.sort(key=lambda ig: ig.count(MISSING_VALUE))
        return subgroups

    def _ambiguous_rgs(self, rg_set):
        ambig_rgs = set([])
        for ig in self.reporting_groups:
            if MISSING_VALUE in ig:
                continue
            ig_supergroups = self._get_supergroups(ig, rg_set)
            reporting_level = ig_supergroups[0].count(MISSING_VALUE)
            matches = list(filter(lambda x: x.count(MISSING_VALUE) == reporting_level, ig_supergroups))
            if len(matches) > 1:
                ambig_rgs.update(matches)
        return ambig_rgs

    def _prep_prediction_inputs(self, X, G):
        n = X.shape[0]
        assert len(G) == n
        G = G.reset_index(drop=True)
        y_pred = np.empty([n, ])
        y_pred[:] = float('nan')
        return X, G, y_pred

    def predict(self, X, G):
        X, G, y_pred = self._prep_prediction_inputs(X, G)
        for _, unique_row in G.drop_duplicates().iterrows():
            ig = tuple(unique_row.values)
            y_pred_ig, idx = self.predict_for(X, G, ig, use_missing_val=True)
            assert len(idx)
            y_pred[idx] = y_pred_ig
        assert not np.isnan(y_pred).any()
        return y_pred

    def predict_for(self, X, G, ig=None, use_missing_val=False):
        """
        Get y predictions given X and G
        :param X: numpy.ndarray
        :param G: pandas.core.frame.DataFrame
        :return: numpy.ndarray
        """
        X, G, _ = self._prep_prediction_inputs(X, G)
        ig_f = self.ig_filtered_map[ig]
        h = self.models[self.assignments[ig_f]['model_index']]
        return h.predict_for(X, G, ig, use_missing_val)

    def predict_proba(self, X, G):
        """
        Get y predictions given X and G
        :param X: numpy.ndarray
        :param G: pandas.core.frame.DataFrame
        :return: numpy.ndarray
        """
        X, G, y_pred_proba = self._prep_prediction_inputs(X, G)
        for _, unique_row in G.drop_duplicates().iterrows():
            ig = tuple(unique_row.values)
            y_pred_proba_ig, idx = self.predict_proba_for(X, G, ig, use_missing_val=True)
            assert len(idx)
            y_pred_proba[idx] = y_pred_proba_ig
        assert not np.isnan(y_pred_proba).any()
        return y_pred_proba

    def predict_proba_for(self, X, G, ig, use_missing_val=False):
        """
        Get y predictions given X and G
        :param X: numpy.ndarray
        :param G: pandas.core.frame.DataFrame
        :return: numpy.ndarray
        """
        X, G, _ = self._prep_prediction_inputs(X, G)
        ig_f = self.ig_filtered_map[ig]
        h = self.models[self.assignments[ig_f]['model_index']]
        return h.predict_proba_for(X, G, ig, use_missing_val)


class FlatSystem(PersonalizedSystem):
    def __init__(self, data, generic_model, assignment_metric='auc', assignment_sample='training', **kwargs):
        """
        Prediction System given all group attributes are reported at the same time
        :param data: BinaryClassificationDataset
        :param generic_model: ClassificationModel
        :param assignment_metric: string (e.g. 'auc')
        """
        super().__init__(data=data, generic_model=generic_model, metric_name=assignment_metric,
                         assignment_sample=assignment_sample,
                         delta=kwargs.get('delta', 0.0))
        self._tikz_repr = ""
        self._tikz_repr_pruned = ""

    @property
    def required_group_attributes(self):
        """dictionary containing the group attributes that are required to predict with each group's classifier"""
        return {ig: self.models[self.assignments[ig_a]['model_index']].required_group_attributes for ig, ig_a in
                self.ig_filtered_map.items()}

    def _reset_vars(self):
        self._tikz_repr = ""
        self._tikz_repr_pruned = ""
        super()._reset_vars()

    def _set_ig_filtered_map(self):
        """
        Create map from reporting group to filtered reporting group,
        such that the filtered reporting group contains only data
        used by the model
        :return:
        """
        self._ig_filtered_map = {ig: ig for ig in self.reporting_groups}
        for ig, info in self.assignments.items():
            if ig != self.dnr_all_group and info['model_index'] == 0:
                self._ig_filtered_map[ig] = self.dnr_all_group

    def to_tikz(self, include_stats=True, viz_pruned=True):
        """ tikz  representation of the system for latex """
        if viz_pruned and self._tikz_repr_pruned:
            return self._tikz_repr_pruned
        elif not viz_pruned and self._tikz_repr:
            return self._tikz_repr
        size = .33
        tikz_str = ""
        for ig, info in self.assignments.items():
            h_desc = f"""h_{{{info['model_index']}}}"""
            pruned = self.ig_filtered_map[ig] != ig and viz_pruned
            if pruned:
                model_type_str = "prunednode"
            elif info['model_index'] == 0:
                model_type_str = f"""genericnode={{{HUE}}}"""
            else:
                model_type_str = f"""personnode={{{HUE}}}{{{size}em}}"""
            leaf_label = f"""[ , leaflabel={{{group_name_latex(ig)}}}]"""
            edge_label_str = "grouplabelpruned={" if pruned else "grouplabel={"
            edge_label_str += f"""\\userstatssimple"""
            edge_label_str += f"""{{{info['gain_over_generic']:.3f}}}{{{info['prob_change_over_generic'] * 100:.0f}}}"""
            edge_label_str += "}"
            if include_stats:
                hue = HUE if not pruned else 0
                box_color = f"{H_COLOR}!{hue}" if (info['model_index'] != 0) else f"lightgray!{hue}"
                node_table_str = f"""\\nodestats{{{box_color}}}"""
                for metric_name in METRIC_NAMES:
                    node_table_str += f"""{{\\noderow{{{metric_name.replace('log_', '').upper()}}}"""
                    for sample_type in ['training', 'validation', 'test']:
                        perf = self.performance_map[info['model_index']][ig][sample_type]['perf']
                        node_table_str += f"""{{{metric_latex(perf, metric_name)}}}"""
                    node_table_str += "}"
                ig_tikz_str = ("\\begin{forest}\n"
                               f"""[,[\\nodemodel{{${h_desc}$}}\\\\{node_table_str},{model_type_str},{edge_label_str}{leaf_label}\n]]"""
                               "\\end{forest}\n")
                tikz_str += ig_tikz_str
            else:
                raise NotImplementedError
        flat_tex = "\\begin{adjustbox}{width=1.0\linewidth,center}\n" + tikz_str + "\\end{adjustbox}"
        flat_tex = clean_latex_str(flat_tex)
        # check that all brackets/parentheses match
        assert valid_tikz(flat_tex)
        if viz_pruned:
            self._tikz_repr_pruned = flat_tex
        else:
            self._tikz_repr = flat_tex
        return flat_tex


class SequentialSystem(PersonalizedSystem):
    def __init__(self, data, generic_model, assignment_metric='auc', assignment_sample='training',
                 criteria='min_leaves',
                 **kwargs):
        """
        Prediction System given all group attributes can be reported sequentially
        :param data: BinaryClassificationDataset
        :param generic_model: ClassificationModel
        :param assignment_metric: string (e.g. 'auc')
        :param criteria: string (criteria for choosing tree)
        """
        # general
        super().__init__(data=data, generic_model=generic_model, metric_name=assignment_metric,
                         assignment_sample=assignment_sample,
                         delta=kwargs.get('delta', 0.0))

        self.criteria = criteria
        self.label_map = self.data.group_attributes.labels

        self.all_trees = []
        self.trees_pruned = False
        self.best_tree = None
        self.min_sample_size = kwargs.get('min_sample_size', 0)
        self._tikz_strs = {}
        self._flattened_reporting_options = None
        self._flat_tikz_repr = ""

    @property
    def required_group_attributes(self):
        """dictionary containing the group attributes that are required to predict with each group's classifier"""
        return {ig: self.models[self.assignments[ig_a]['model_index']].required_group_attributes for ig, ig_a in
                self.ig_filtered_map.items()}

    @property
    def all_trees_img(self):
        all_trees_imgs = [t.to_img() for t in self.all_trees]
        max_h = max([img.size[1] for img in all_trees_imgs])
        total_w = sum([img.size[0] for img in all_trees_imgs])
        composite_image = Image.new('RGB', (total_w, max_h), (255, 255, 255))
        x = 0
        for t_img in all_trees_imgs:
            composite_image.paste(t_img, (x, 0))
            x += t_img.size[0]
        return composite_image

    @property
    def flattened_reporting_options(self):
        if self._flattened_reporting_options is None:
            self._get_flattened_reporting_options()
        return self._flattened_reporting_options

    def _reset_vars(self):
        self.all_trees = []
        self.trees_pruned = False
        self.best_tree = None
        self._tikz_strs = {}
        super()._reset_vars()

    def generate_all_trees(self):
        """
        Generate all possible trees given attributes
        :return: None
        """
        self.all_trees = self._generate_all_trees_helper(list(self.label_map.keys()))

    def prune_all_trees(self):
        """
        Prune all trees to avoid asking for unhelpful personal data
        :return:
        """
        assert len(self.all_trees)
        for t in self.all_trees:
            self.prune(t)
            t.assignment_metric = self.metric_name
        self.trees_pruned = True
        self.set_best_tree()

    def set_best_tree(self, ind=None):
        """
        Set the best tree after pruning based on criteria or index
        :param ind:
        :return:
        """
        if not len(self.all_trees):
            self.generate_all_trees()
        if not self.trees_pruned:
            self.prune_all_trees()
        if ind is not None:
            self.best_tree = self.all_trees[ind]
        elif self.criteria == 'min_leaves':
            self.best_tree = min(self.all_trees, key=lambda t: getattr(t, 'n_leaves'))
        else:
            raise NotImplementedError()

    def predict(self, X, G):
        """
        Get y predictions given X and G
        :param X: numpy.ndarray
        :param G: pandas.core.frame.DataFrame
        :return: numpy.ndarray
        """
        if self.best_tree is None:
            self.set_best_tree()
        return super().predict(X, G)

    def predict_proba(self, X, G):
        """
        Get y predictions given X and G
        :param X: numpy.ndarray
        :param G: pandas.core.frame.DataFrame
        :return: numpy.ndarray
        """
        if self.best_tree is None:
            self.set_best_tree()
        return super().predict_proba(X, G)

    def _set_ig_filtered_map(self):
        """
        Create map from reporting group to filtered reporting group,
        such that the filtered reporting group contains only data
        used by the model
        :return:
        """
        if self.best_tree is None:
            self.set_best_tree()
        self._ig_filtered_map = {}
        for ig in self.reporting_groups:
            curr = self.best_tree.root
            while not curr.is_leaf:
                match = None
                for c in filter(lambda c: not c.pruned and c.g != MISSING_VALUE, curr.children):
                    if is_subgroup(ig, c.ig):
                        match = c
                if match is None:
                    match = next(filter(lambda c: c.g == MISSING_VALUE, curr.children))
                curr = match
            self._ig_filtered_map[ig] = curr.ig

    def _generate_all_trees_helper(self, attrs):
        """
        Recursive helper function for generating all possible trees
        :param attrs:
        :return:
        """
        if len(attrs) == 0:
            return []
        if len(attrs) == 1:
            a = attrs[0]
            n = Node(name=a, g='', pruned=False, gain_p=float('nan'), gain_r=float('nan'),
                     model_index=0, ig='')
            children = [
                Node(g=g, parent=n, name='', pruned=False, gain_p=float('nan'), gain_r=float('nan'),
                     model_index=0, ig='')
                for g in self.label_map[a] + [MISSING_VALUE]]
            n.children = children
            return [PersonalizationTree(n)]
        trees = []
        for i, a in enumerate(attrs):
            rem_attrs = attrs[:i] + attrs[i + 1:]
            sub_trees = self._generate_all_trees_helper(rem_attrs)
            # get all possible subtrees using all possible values + 1 (MISSING_VALUE)
            sub_combs = list(product(sub_trees, repeat=len(self.label_map[a] + [MISSING_VALUE])))
            for sc in sub_combs:
                n = Node(name=a, g='', pruned=False, gain_p=float('nan'), gain_r=float('nan'),
                         model_index=0, ig='')
                children = [Node(name=t.root.name, g=g, children=deepcopy(t.root.children), parent=n, pruned=False,
                                 gain_p=float('nan'), gain_r=float('nan'), model_index=0, ig='')
                            for g, t in
                            zip(self.label_map[a] + [MISSING_VALUE], sc)]
                n.children = children
                trees.append(PersonalizationTree(n))
        return trees

    def to_tikz(self, t, include_stats=True):
        """ tikz forest representation of the tree for latex """
        if t in self._tikz_strs:
            return self._tikz_strs[t]
        tikz_str = self._dfs_tikz_helper(t.root, include_stats)
        # make latex friendly, e.g. remove unicode
        tikz_str = clean_latex_str(tikz_str)
        # check that all brackets/parentheses match
        assert valid_tikz(tikz_str)
        tree_tex = ("\\begin{adjustbox}{width=1.0\linewidth,center}\n"
                    "\\begin{forest}\n"
                    "forked edges,\n" + tikz_str + "\n"
                                                   "\\end{forest}\n"
                                                   "\\end{adjustbox}")
        self._tikz_strs[t] = tree_tex
        return self._tikz_strs[t]

    def _get_flattened_reporting_options(self):
        """ flatten trees to non-redundant reporting groups """
        # get all leaf sets
        leaf_sets = [set([l.ig for l in t.leaves if (l.gain_r) or len(set(l.ig)) == 1]) for t in self.all_trees]
        flat_set = set.intersection(*leaf_sets)
        # identify ambiguity (a group has 2+ "most specific" options)
        ambig_rgs = self._ambiguous_rgs(flat_set)
        candidate_sets = []
        for ambig_rg in ambig_rgs:
            for leaf_set in leaf_sets:
                if all([is_subgroup(ig, ambig_rg) for ig in leaf_set - flat_set]) and len(leaf_set - flat_set):
                    cs = leaf_set.copy()
                    cs.remove(ambig_rg)
                    candidate_sets.append(cs)
        flat_set = min(candidate_sets, key=lambda cs: len(cs)) if candidate_sets else flat_set
        self._flattened_reporting_options = flat_set

    def to_flattened_tikz(self, include_stats=True):
        """Tikz for "optimized" flat system"""
        if self._flat_tikz_repr:
            return self._flat_tikz_repr
        size = .33
        tikz_str = ""
        for ig, info in self.assignments.items():
            h_desc = f"""h_{{{info['model_index']}}}"""
            pruned = ig not in self.flattened_reporting_options
            if pruned:
                model_type_str = "prunednode"
            elif info['model_index'] == 0:
                model_type_str = f"""genericnode={{{HUE}}}"""
            else:
                model_type_str = f"""personnode={{{HUE}}}{{{size}em}}"""
            if pruned:
                leaf_label = f"""[ , leaflabelpruned={{{group_name_latex(ig)}}}]"""
            else:
                leaf_label = f"""[ , leaflabel={{{group_name_latex(ig)}}}]"""
            if include_stats:
                hue = HUE if not pruned else 0
                box_color = f"{H_COLOR}!{hue}" if (info['model_index'] != 0) else f"lightgray!{hue}"
                node_table_str = f"""\\nodestats{{{box_color}}}"""
                for metric_name in METRIC_NAMES:
                    node_table_str += f"""{{\\noderow{{{metric_name.replace('log_', '').upper()}}}"""
                    for sample_type in ['training', 'validation', 'test']:
                        perf = self.performance_map[info['model_index']][ig][sample_type]['perf']
                        node_table_str += f"""{{{metric_latex(perf, metric_name)}}}"""
                    node_table_str += "}"
                ig_tikz_str = ("\\begin{forest}\n"
                               f"""[\\nodemodel{{${h_desc}$}}\\\\{node_table_str},{model_type_str},{leaf_label}\n]"""
                               "\\end{forest}\n")
                tikz_str += ig_tikz_str
            else:
                raise NotImplementedError
        flat_tex = "\\begin{adjustbox}{width=1.0\linewidth,center}\n" + tikz_str + "\\end{adjustbox}"
        assert valid_tikz(flat_tex)
        self._flat_tikz_repr = flat_tex
        return self._flat_tikz_repr

    def _dfs_tikz_helper(self, n, include_stats, level=0):
        if not n:
            return ''
        # add a node for the splitting name (e.g., sex)
        suffix = ''
        h_desc = f"""h_{{{n.model_index}}}"""
        n_desc = self.tikz_node_str(n, level, h_desc, include_stats=include_stats)
        tabs = '\t' * level
        if n.name:
            suffix = ']'
            tabs = '\t' * level
            n_desc = f"""{n_desc}\n{tabs}\t[{n.name.lower()}?, groupattr"""
        c_desc = ''.join([self._dfs_tikz_helper(c, level + 1) for c in n.children])
        return f"""\n{tabs}[{n_desc}{c_desc}]{suffix}"""

    def compute_node_stats(self, t):
        for n in t.nodes:
            ig_path = path_between(t.root, n)
            ig = path_to_ig(ig_path, self.name_index_map)
            n.ig = ig
            n.model_index = self.assignments[n.ig]['model_index']
            n_perf = self.performance_map[n.model_index][n.ig][self._assignment_sample]['perf']
            n_sample_size = self.performance_map[n.model_index][n.ig][self._assignment_sample]['sample_size']
            n_preds = self.performance_map[n.model_index][n.ig][self._assignment_sample]['preds']
            r_perf = self.performance_map[0][n.ig][self._assignment_sample]['perf']
            r_preds = self.performance_map[0][n.ig][self._assignment_sample]['preds']
            n.gain_r = raw_gain(r_perf, n_perf, self.metric_name)
            n.change_r = np.sum(np.not_equal(n_preds, r_preds)) / len(n_preds)
            if n.parent:
                p_perf = self.performance_map[n.parent.model_index][n.ig][self._assignment_sample]['perf']
                p_preds = self.performance_map[n.parent.model_index][n.ig][self._assignment_sample]['preds']
                n.gain_p = raw_gain(p_perf, n_perf, self.metric_name)
                n.change_p = np.sum(np.not_equal(n_preds, p_preds)) / len(n_preds)
            else:
                n.change_p = n_sample_size
                n.gain_p = float('nan')
        n = t.root
        while n.children:
            n = list(filter(lambda c: c.g == MISSING_VALUE, n.children, ))[0]
        t.node_stats_assigned = True

    def tikz_node_str(self, n, level, h_desc, include_stats=True):
        l_hue = 0 if n.pruned else min(80, HUE * (level + 1))
        size = .33
        if n.pruned:
            model_type_str = "prunednode"
        elif n.model_index == 0:
            model_type_str = f"""genericnode={{{l_hue}}}"""
        else:
            model_type_str = f"""personnode={{{l_hue}}}{{{size}em}}"""
        edge_label_str = "grouplabelpruned={" if n.pruned else "grouplabel={"
        # don't include future stats for leaf nodes
        if n.is_leaf:
            edge_label_str += f"""\\userstatsleaf{{{clean_latex_str(n.g)}}}"""
            edge_label_str += f"""{{{n.gain_r:.3f}}}{{{n.gain_p:.3f}}}"""
            edge_label_str += f"""{{{n.change_r * 100:.0f}}}{{{n.change_p * 100:.0f}}}"""
        else:
            descendants = list(filter(lambda d: not d.pruned, n.descendants))
            gain_f = max([d.gain_r for d in descendants], key=lambda x: abs(x)) if len(descendants) else n.gain_r
            change_f = max([d.change_r for d in descendants], key=lambda x: abs(x)) if len(descendants) else n.change_r
            edge_label_str += f"""\\userstats{{{clean_latex_str(n.g)}}}"""
            edge_label_str += f"""{{{n.gain_r:.3f}}}{{{n.gain_p:.3f}}}{{{gain_f:.3f}}}"""
            edge_label_str += f"""{{{n.change_r * 100:.0f}}}{{{n.change_p * 100:.0f}}}{{{change_f * 100:.0f}}}"""
        edge_label_str += "}"
        leaf_label = f"""[ , leaflabel={{{group_name_latex(n.ig)}}}]""" if (n.is_leaf and not n.pruned) else ""
        if include_stats:
            box_color = f"{H_COLOR}!{l_hue}" if (n.model_index != 0) else f"lightgray!{l_hue}"
            node_table_str = f"""\\nodestats{{{box_color}}}"""
            for metric_name in METRIC_NAMES:
                node_table_str += f"""{{\\noderow{{{metric_name.replace('log_', '').upper()}}}"""
                for sample_type in ['training', 'validation', 'test']:
                    perf = self.performance_map[n.model_index][n.ig][sample_type]['perf']
                    node_table_str += f"""{{{metric_latex(perf, metric_name)}}}"""
                node_table_str += "}"
            return f"""\\nodemodel{{${h_desc}$}}\\\\{node_table_str},{model_type_str},{edge_label_str}{leaf_label}"""
        else:
            raise NotImplementedError

    def prune(self, t):
        """
        Prune leaves that do not lead to gain
        :param t: tree to prune
        :return: None
        """
        t.node_stats_assigned = False
        self.compute_node_stats(t)
        # reset all nodes
        for n in t.all_nodes:
            n.pruned = False
        stack = t.leaves.copy()
        while stack:
            leaf = stack.pop()
            # users should always retain the ability to withhold info
            if leaf.g == MISSING_VALUE:
                continue
            # if no samples for leaf, should prune
            parent = leaf.parent
            p_i, l_i = parent.model_index, leaf.model_index
            sample_size = self.performance_map[l_i][leaf.ig][self._assignment_sample]['sample_size']
            insufficient_samples = sample_size < self.min_sample_size
            if p_i == l_i or self.check_fair_use(p_i, leaf.ig) and (
                    insufficient_samples or not self.update_rule(p_i, l_i, leaf.ig)):
                if insufficient_samples:
                    print('INSUFFICIENT SAMPLES FOR ', leaf.ig, ', PRUNING')
                leaf.pruned = True
                remaining_children = list(filter(lambda pc: not pc.pruned, list(parent.children)))
                # if we pruned the last child, prune the N/A option
                if len(remaining_children) == 1 and remaining_children[0].g == MISSING_VALUE:
                    remaining_children[0].pruned = True
                    stack.append(parent)
                if not len(remaining_children):
                    stack.append(parent)


class PersonalizationTree(object):
    """ Tree in which nodes split by group attribute and leaves represent reporting groups """

    def __init__(self, root):
        """
        PersonalizationTree object
        :param root: Node
        """
        self.root = root
        self.node_stats_assigned = False
        self.assignment_metric = ''

    #### built-ins ####
    def __repr__(self):
        return str(RenderTree(self.pruned_tree).by_attr(lambda n: node_desc(n)))

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return id(str(self))

    #### properties ####
    @property
    def tree_with_stats_str(self):
        """ print tree including node stats """
        assert self.node_stats_assigned, "Node stats not assigned, use method assign_node_stats"
        return str(RenderTree(self.pruned_tree).by_attr(
            lambda n: node_desc(n, include_stats=True, metric_name=self.assignment_metric)))

    @property
    def all_nodes(self):
        """ all nodes, including those that are pruned """
        return list(PreOrderIter(self.root))

    @property
    def all_leaves(self):
        """ all leaves, including those that are pruned """
        return list(filter(lambda node: node.is_leaf, self.all_nodes))

    @property
    def nodes(self):
        """ all nodes"""
        return [node for node in self.all_nodes if not node.pruned]

    @property
    def leaves(self):
        """ all leaf nodes"""
        return list(filter(lambda n: np.all([c.pruned for c in n.children]), self.nodes))

    @property
    def n_leaves(self):
        """number of leaves"""
        return len(self.leaves)

    @property
    def depth(self):
        """ maximum path length """
        return max([len(p) for p in self.root_to_leaf_paths])

    def to_img(self, include_stats=False):
        """ Image representation of tree """
        # we need a unicode-friendly font to create the image
        font_path = '/Library/Fonts/Arial Unicode.ttf'
        assert os.path.exists(font_path)
        # set font and font size
        fnt = ImageFont.truetype(font_path, 50)
        # set the width of the image based on tree depth and label length
        width = 310 * self.depth if include_stats else 150 * self.depth
        # height based on number of lines
        height = str(self).count('\n') * 65
        # add extra height if including stats
        height += 200 * self.n_leaves if include_stats else 0
        # white background
        img = Image.new('RGB', (width, height), (255, 255, 255))
        d = ImageDraw.Draw(img)
        str_rep = self.tree_with_stats_str if include_stats else str(self)
        # black text
        d.text((50, 0), str_rep, fill=(0, 0, 0), font=fnt)
        return img

    def raw_tree_str(self):
        """ print tree, ignoring pruned nodes """
        return str(RenderTree(self.root).by_attr(lambda n: node_desc(n)))

    #### paths ####
    def path_to(self, node):
        """
        returns path to node in tree
        """
        return path_between(self.root, node)

    @property
    def root_to_leaf_paths(self):
        """ list of paths from root to each leaf"""
        return [self.path_to(l) for l in self.leaves]

    @property
    def pruned_tree(self):
        """ pruned tree """
        root = deepcopy(self.root)
        all_nodes = list(PreOrderIter(root))
        for n in all_nodes:
            if n.children:
                n.children = list(filter(lambda c: not c.pruned, list(n.children)))
        return root


### TREE UTILS

def path_to_ig(path, name_index_map):
    """ convert path (list of nodes) to reporting group tuple """
    if np.any([isinstance(el, Iterable) for el in path]):
        path = flatten_path(path)
    intersectional_group = [MISSING_VALUE] * len(name_index_map)
    for i in range(len(path) - 1):
        intersectional_group[name_index_map[path[i].name]] = path[i + 1].g
    return tuple(intersectional_group)


def flatten_path(nested_path):
    """ flatten path from Walker() to generator of nodes """
    for el in list(nested_path):
        if isinstance(el, Iterable):
            for sub in flatten_path(el):
                yield sub
        else:
            yield el


def path_between(n1, n2):
    """
    Path between two nodes
    :param n1: Node
    :param n2: Node
    :return:
    """
    w = Walker()
    return list(flatten_path(w.walk(n1, n2)))


def node_str(n):
    return f"\n{n.name.upper()}" if n.name and len(n.children) else ""


def node_desc(n, include_stats=False, metric_name="perf"):
    """
    Create string description for a node (useful for printing)
    :param n: Node
    :param include_stats: include performance statistics (node perf, parent perf, sample size)
    :param metric_name: tikz_str (e.g., log loss)
    :return:
    """
    n_stats = ''
    if include_stats and not n.children:
        return node_stats_str(n, metric_name)
    return n.g + n_stats + node_str(n)


def node_stats_str(n, metric_name):
    n_stats = f"\n{metric_name} gain over prev: {n.gain_p:.4f}\n" \
              f"{metric_name} gain over root: {n.gain_r:.4f}"
    return n_stats


def valid_tikz(tikz_str):
    stack = []
    openers = {'{', '[', '('}
    closers = {']': '[', '}': '{', ')': '('}
    for c in tikz_str:
        if c in openers:
            stack.append(c)
        elif c in closers:
            if not stack:
                return False
            o = stack.pop()
            if o != closers[c]:
                return False
    return len(stack) == 0
