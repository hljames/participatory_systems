"""
this file contains an auditor class to compute performance statistics
to check the gains of personalization over intersectional groups
"""
from sklearn.utils import resample
from psc.data import *
from psc import metrics
from psc.tests.test_hypothesis_testing import two_sample_paired_sign_test
from psc.utils import METRIC_NAMES, MODEL_TYPES, clean_latex_str, product, PARTIAL_REPORTING_MODEL_TYPES
from psc.groups import MISSING_VALUE


class PersonalizationAuditor(object):
    """
    This object contains convenience functions to test fair use guarantees
    """

    def __init__(self, generic_model, data, n_bootstrap=30, **kwargs):
        """
        :param data: BinaryClassificationDataset
        :param kwargs:
        """
        self.generic_model = generic_model
        self.data = data
        self.intersectional_groups = data.group_encoder.groups
        self.intersectional_groups_names = data.group_encoder.dummy_names
        self.n_bootstrap = n_bootstrap

        # generate indices for bootstrap resampling
        self.random_state = np.random.RandomState(kwargs.get('random_seed', 1337))
        self.bootstrap_indices = {}
        self.data_dict = {'data_name': kwargs.get('data_name', ''),
                          'rebalancing_type': kwargs.get('rebalancing_type', ''),
                          'fold_id': self.data.fold_id,
                          'method_name': kwargs.get('method_name', ''),
                          'assignment_metric': kwargs.get('assignment_metric', None),
                          'assignment_sample': kwargs.get('assignment_sample', None),
                          'fold_num_validation': self.data.fold_num_validation,
                          'fold_num_test': self.data.fold_num_test
                          }
        for sample_type in data.SAMPLE_TYPES:
            sample = getattr(self.data, sample_type)
            self.bootstrap_indices[sample_type] = self.generate_bootstrap_indices(y=sample.y, G=sample.G,
                                                                                  n_bootstrap=self.n_bootstrap)

    def generate_bootstrap_indices(self, y, G, n_bootstrap=1000):
        """
        generates indices for bootstrap resampling
        bootstrap indices are generated using stratified sampling to ensure that the
        the prevalence and relative size of each group is the same across bootstrap samples
        :param y: labels
        :param G: data frame of group attributes
        :param n_bootstrap: number of bootstrap indices required
        :return:
        """
        # combine y values with group info (e.g., -1 Male Old, +1 Female Young, ...)
        yG = np.hstack((y[:, np.newaxis], np.array(G.values, dtype=str)))
        # integer group value for each y entry
        _, yg_type = np.unique(yG, axis=0, return_index=False, return_inverse=True)
        # n_bootstrap x len dataset
        bootstrap_indices = np.tile(np.nan, (n_bootstrap, len(y)))
        # g is integer index of group (e.g., -1 Female Old)
        for g in np.unique(yg_type):
            # all inds of group g
            yg_idx = np.flatnonzero(np.isin(yg_type, g))
            # generate random sample of indicies from group g
            # random_state=None to avoid reproducing same sample each time
            yg_bootstrap_idx = np.array(
                [resample(yg_idx, random_state=None) for i in range(n_bootstrap)])
            # set bootstrap indicies for current g
            bootstrap_indices[:, yg_idx] = yg_bootstrap_idx
        return bootstrap_indices.astype(int)

    def compute_required_attributes(self, model):
        """
        compute the required group attributes for generic or personalized model for all intersectional groups
        :param model: model for computing performance
        :param model_type: 'generic' or 'personalized'
        :param metric_name: 'err', 'ece', 'auc'
        :param sample_type: 'training', 'validation', 'test'
        :return: vector of performance metrics for each intersectional group
        """
        req_attrs = model.required_group_attributes
        if isinstance(req_attrs, dict):
            values = np.repeat(None, len(self.intersectional_groups))
            for k, group in enumerate(self.intersectional_groups):
                values[k] = req_attrs[group]
        else:
            values = np.repeat(req_attrs, len(self.intersectional_groups))
        return values

    def compute_data_use(self, model, model_type='generic', metric_name='ece', sample_type='training'):
        """
        compute the performance for generic or personalized model for all intersectional groups
        :param model: model for computing performance
        :param model_type: 'generic' or 'personalized'
        :param metric_name: 'err', 'ece', 'auc'
        :param sample_type: 'training', 'validation', 'test'
        :param return_type: 'dataframe'
        :param include_all_group: include group 'all'
        :return: vector of performance metrics for each intersectional group
        """
        # select function to compute metric and provide predictions
        assert model_type in MODEL_TYPES
        assert metric_name in METRIC_NAMES

        # specify data sample
        sample = getattr(self.data, sample_type)
        y = sample.y
        X = sample.X
        G = sample.G.reset_index(drop=True)

        reporting_options = {k: v + [MISSING_VALUE] for k, v in self.data.group_attributes.labels.items()}
        reporting_groups = list(product(*list(reporting_options.values())))
        predict_generic_for = getattr(self.generic_model, f'{metrics.predict_handles_map[metric_name]}_for')

        group_name_1 = []
        feasible_reporting_option = []
        is_rational_leaf = []
        n_informed_consent = []
        n_1 = []
        n_reportable_attrs = []
        n_requested_attrs = []
        if model_type in PARTIAL_REPORTING_MODEL_TYPES:
            unpruned_rgs = set(model.ig_filtered_map.values())
            full_reporting_groups = model.full_reporting_groups
        else:
            unpruned_rgs = full_reporting_groups = set(self.data.group_encoder._intersectional_labels)
        for rg in reporting_groups:
            group_name_1.append(rg)
            feasible_reporting_option.append(rg in unpruned_rgs)
            is_rational_leaf.append(rg in full_reporting_groups)
            if model_type in PARTIAL_REPORTING_MODEL_TYPES:
                if rg == model.dnr_all_group:
                    n_informed_consent.append(0)
                elif model_type == 'flat':
                    n_informed_consent.append(0)
                elif rg not in unpruned_rgs:
                    n_informed_consent.append(float('nan'))
                else:
                    n_informed_consent.append(len(rg) - rg.count(MISSING_VALUE))
            elif MISSING_VALUE in rg:
                n_informed_consent.append(float('nan'))
            else:
                n_informed_consent.append(0)
            _, idx = predict_generic_for(X, G, rg)
            n_1.append(len(idx))
            n_reportable_attrs.append(len(rg) - rg.count(MISSING_VALUE))
            if model_type in PARTIAL_REPORTING_MODEL_TYPES:
                rg_f = model.ig_filtered_map[rg]
                n_requested_attrs.append(len(rg_f) - rg_f.count(MISSING_VALUE))
            else:
                n_requested_attrs.append(float('nan') if MISSING_VALUE in rg else len(rg))

        df_dict = self.data_dict.copy()
        df_dict.update({'sample_type': sample_type,
                        'model': model,
                        'model_type': model_type,
                        'group_name_1': group_name_1,
                        'n_1': n_1,
                        'feasible_reporting_option': feasible_reporting_option,
                        'is_rational_leaf': is_rational_leaf,
                        'n_informed_consent': n_informed_consent,
                        'n_reportable_attrs': n_reportable_attrs,
                        'n_requested_attrs': n_requested_attrs})
        return pd.DataFrame(df_dict)

    def compute_performance_metrics(self, model, model_type='generic', metric_name='ece', sample_type='training',
                                    return_type='', include_all_group=True, use_intersectional_groups=False,
                                    all_group_only=False):
        """
        compute the performance for generic or personalized model for all intersectional groups
        :param model: model for computing performance
        :param model_type: 'generic' or 'personalized'
        :param metric_name: 'err', 'ece', 'auc'
        :param sample_type: 'training', 'validation', 'test'
        :param return_type: 'dataframe'
        :param include_all_group: include group 'all'
        :return: vector of performance metrics for each intersectional group
        """
        # select function to compute metric and provide predictions
        assert model_type in MODEL_TYPES
        assert metric_name in METRIC_NAMES

        # specify data sample
        sample = getattr(self.data, sample_type)
        y = sample.y
        X = sample.X
        G = sample.G.reset_index(drop=True)

        # prediction handles
        compute_metric = getattr(metrics, f'compute_{metric_name}')
        predict_personalized = getattr(model, f'{metrics.predict_handles_map[metric_name]}')
        predict_generic = getattr(self.generic_model, f'{metrics.predict_handles_map[metric_name]}')
        y_pred_gen = predict_generic(X=X, G=None)

        # compute predictions on sample
        if model_type in ('generic'):
            y_pred = predict_personalized(X=X, G=None)
        else:
            y_pred = predict_personalized(X=X, G=G)

        if model_type not in PARTIAL_REPORTING_MODEL_TYPES or use_intersectional_groups:
            groups = self.intersectional_groups
            group_to_index = {g: i for i, g in enumerate(groups)}
        else:
            groups = model.full_reporting_groups
            group_to_index = {ig: groups.index(model.ig_filtered_map[ig]) for ig in self.intersectional_groups}
        group_names = list(clean_latex_str(' & '.join(name).lower()) for name in groups)

        # compute performance metric for each group
        values, gen_values, sample_sizes = [], [], []
        group_indices = self.data.group_encoder.to_indices(G, group_to_index)
        # compute metric for each group
        if not all_group_only:
            for k in range(len(groups)):
                idx = np.isin(group_indices, k)
                sample_sizes.append(len(idx))
                if len(idx):
                    values.append(compute_metric(y[idx], y_pred[idx]))
                    gen_values.append(compute_metric(y[idx], y_pred_gen[idx]))
                else:
                    values.append(float('nan'))
                    gen_values.append(float('nan'))
        if include_all_group or all_group_only:
            values.append(compute_metric(y, y_pred))
            gen_values.append(compute_metric(y, y_pred_gen))
            sample_sizes.append(len(y))
            group_names = ['all'] if all_group_only else group_names
        if metric_name != 'auc':
            gain = list(np.array(gen_values) - np.array(values))
        else:
            gain = list(np.array(values) - np.array(gen_values))
        if return_type == 'dataframe':
            df_dict = self.data_dict.copy()
            df_dict.update({'sample_type': sample_type,
                            'model': model,
                            'model_type': model_type,
                            'group_name_1': np.array(group_names),
                            'n_1': sample_sizes,
                            'group_name_2': float('nan'),
                            'n_2': float('nan'),
                            'metric_name': metric_name,
                            'value': values,
                            'value_generic': gen_values,
                            'gain': gain,
                            'pvalue': float('nan'),
                            'tstat': float('nan')})
            return pd.DataFrame(df_dict)
        else:
            return values

    def compute_rationality_metrics(self, model, model_type='generic', metric_name='ece', sample_type='training',
                                    return_pvalues=False, use_intersectional_groups=False, return_type=''):
        """
        compute the gain in personalization for all intersectional groups
        :param model: model for computing performance
        :param metric_name: type of performance metric (e.g., 'ece', 'auc', 'error')
        :param sample_type: sample type ('training', 'validation', 'test')
        :param return_pvalues: run a hypothesis test to produce pvalues
        :return: a dictionary of vectors containing the gains from personzliation, raw_values, p_values, and test statistics for each intersectional group
        """
        assert model_type in MODEL_TYPES
        assert metric_name in METRIC_NAMES

        # specify data sample
        sample = getattr(self.data, sample_type)
        y = sample.y
        X = sample.X
        G = sample.G.reset_index(drop=True)
        if model_type not in PARTIAL_REPORTING_MODEL_TYPES or use_intersectional_groups:
            groups = self.intersectional_groups
            group_to_index = {g: i for i, g in enumerate(groups)}
        else:
            groups = model.full_reporting_groups
            group_to_index = {ig: groups.index(model.ig_filtered_map[ig]) for ig in self.intersectional_groups}
        group_names = list(clean_latex_str(' & '.join(name).lower()) for name in groups)

        # prediction handles
        compute_metric = getattr(metrics, f'compute_{metric_name}')
        predict_personalized = getattr(model, f'{metrics.predict_handles_map[metric_name]}')
        predict_personalized_for = getattr(model, f'{metrics.predict_handles_map[metric_name]}_for')
        predict_generic = getattr(self.generic_model, f'{metrics.predict_handles_map[metric_name]}')
        predict_generic_for = getattr(self.generic_model, f'{metrics.predict_handles_map[metric_name]}_for')

        # compute performance metric for each group
        n_groups = len(groups)
        values = [float('nan')] * len(groups)
        gen_values = [float('nan')] * len(groups)
        sample_sizes = [0] * len(groups)
        for k, rg in enumerate(groups):
            y_pred_rg, idx = predict_personalized_for(X, G, rg, use_missing_val=False)
            gen_y_pred_rg, idx = predict_generic_for(X, G, rg, use_missing_val=False)
            if len(idx):
                sample_sizes[k] = len(idx)
                values[k] = compute_metric(y[idx], y_pred_rg)
                gen_values[k] = compute_metric(y[idx], gen_y_pred_rg)

        # run hypothesis test to check fair use violation via bootstrap resampling
        pvalue = np.repeat(np.nan, n_groups)
        tstat = np.repeat(np.nan, n_groups)
        if return_pvalues:

            IB = self.bootstrap_indices[sample_type]
            yB = y[IB]
            IB_flat = IB.flatten()
            HB_generic = np.reshape(predict_generic(X=X[IB_flat, :], G=None), newshape=IB.shape)
            GB = pd.concat([G.iloc[ib, :] for ib in IB]).reset_index(drop=True)
            group_indices_IB = self.data.group_encoder.to_indices(GB, group_to_index).reshape(IB.shape)
            HB_personalized = np.reshape(predict_personalized(X=X[IB_flat, :], G=GB), newshape=IB.shape)
            # compute metric for each group
            for k in range(len(groups)):
                # nan when there are no samples of group k in bootstrap sample ib
                S_generic, S_personalized = np.repeat(np.nan, IB.shape[0]), np.repeat(np.nan, IB.shape[0])
                for ib, gi in enumerate(group_indices_IB):
                    idx = np.isin(gi, k)
                    if any(idx):
                        S_generic[ib] = compute_metric(yB[ib, idx], HB_generic[ib, idx])
                        S_personalized[ib] = compute_metric(yB[ib, idx], HB_personalized[ib, idx])
                if metric_name != 'auc':
                    S1, S2 = S_generic, S_personalized
                else:
                    S1, S2 = S_personalized, S_generic
                # pvalue is probability of data given rationality violation
                # S1 - S2 should be positive for positive gain
                pvalue[k], tstat[k] = two_sample_paired_sign_test(S1=S1, S2=S2,
                                                                  alternative='less')
        if return_type == 'dict':
            out = {
                'gaps': list(np.array(values) - np.array(gen_values)),
                'raw_values_generic': gen_values,
                'raw_values_personalized': values,
                'pvalue': pvalue,
                'tstat': tstat,
            }
            return out
        if metric_name != 'auc':
            gain = list(np.array(gen_values) - np.array(values))
        else:
            gain = list(np.array(values) - np.array(gen_values))
        df_dict = self.data_dict.copy()
        df_dict.update({'sample_type': sample_type,
                        'model': model,
                        'model_type': model_type,
                        'group_name_1': group_names,
                        'n_1': sample_sizes,
                        'group_name_2': float('nan'),
                        'n_2': float('nan'),
                        'metric_name': metric_name,
                        'value': values,
                        'value_generic': gen_values,
                        'gain': gain,
                        'pvalue': pvalue,
                        'tstat': tstat})
        return pd.DataFrame(df_dict)

    def compute_envyfreeness_metrics(self, model, model_type='generic', metric_name='ece', sample_type='training',
                                     return_pvalues=False, use_intersectional_groups=False, return_type=''):
        """
        :param metric_name:
        :param sample_type:
        :param return_pvalues_and_stats:
        :return:
        """
        # select function to compute metric and provide predictions
        assert model_type in MODEL_TYPES
        assert metric_name in METRIC_NAMES
        data = self.data

        # specify data sample
        sample = getattr(data, sample_type)
        y = sample.y
        X = sample.X
        G = sample.G.reset_index(drop=True)

        if model_type not in PARTIAL_REPORTING_MODEL_TYPES or use_intersectional_groups:
            groups = self.intersectional_groups
            group_to_index = {g: i for i, g in enumerate(groups)}
        else:
            groups = model.full_reporting_groups
            group_to_index = {ig: groups.index(model.ig_filtered_map[ig]) for ig in self.intersectional_groups}
        group_names = list(clean_latex_str(' & '.join(name).lower()) for name in groups)

        # prediction handles
        compute_metric = getattr(metrics, f'compute_{metric_name}')
        predict_personalized = getattr(model, f'{metrics.predict_handles_map[metric_name]}')

        # compute performance of classifier of each group when they report different options for group membership
        n_groups = len(groups)
        raw_values_true = np.tile([np.nan], [n_groups, n_groups])
        raw_values_reported = np.tile([np.nan], [n_groups, n_groups])
        pvalue = np.tile([np.nan], [n_groups, n_groups])
        tstat = np.tile([np.nan], [n_groups, n_groups])
        sample_size = [0] * len(groups)

        # compute gaps
        H_personalized = predict_personalized(X, G)
        group_indices = data.group_encoder.to_indices(G, group_to_index)
        for k, group in enumerate(groups):
            idx = np.isin(group_indices,
                          k)  # boolean of data group membership for group k
            sample_size[k] = sum(idx)
            if not np.any(idx):
                continue
            k_report_true_perf = compute_metric(y[idx],
                                                H_personalized[idx])
            raw_values_true[k, :] = k_report_true_perf
            for l, group_reported in enumerate(groups):
                if group_reported != group:
                    G_reported = pd.DataFrame([group_reported] * np.sum(idx),
                                              columns=G.columns)  # DataFrame of Group Attribute
                    H_reported = predict_personalized(
                        X=X[idx], G=G_reported)  # New Personalized Predictions
                    # Metric under Personalized Predictions
                    raw_values_reported[k, l] = compute_metric(y[idx], H_reported)

        if return_pvalues:
            # run hypothesis test to check fair use violation via bootstrap resampling
            IB = self.bootstrap_indices[sample_type]
            IB_flat = IB.flatten()

            yB = y[IB]
            XB = X[IB_flat, :]
            GB = pd.concat([G.iloc[ib, :] for ib in IB]).reset_index(drop=True)
            group_indices_IB = self.data.group_encoder.to_indices(GB, group_to_index).reshape(IB.shape)

            HB_personalized = np.reshape(
                predict_personalized(X=XB, G=GB), newshape=IB.shape)
            for k, group in enumerate(groups):
                idx = group_indices_IB == k

                S_true = [compute_metric(yb[idxb], hb[idxb]) for yb, hb, idxb in
                          zip(yB, HB_personalized, idx)]

                for l, group_reported in enumerate(groups):
                    if group_reported != group:
                        G_l = pd.DataFrame([group_reported] * len(IB_flat),
                                           columns=self.data.G.columns)
                        H_reported_l = np.reshape(predict_personalized(X=XB, G=G_l),
                                                  newshape=IB.shape)

                        S_misreport = [compute_metric(yb[idxb], hb[idxb]) for yb, hb, idxb in
                                       zip(yB, H_reported_l, idx)]

                        if metric_name != 'auc':
                            S1, S2 = S_misreport, S_true
                        else:
                            S1, S2 = S_true, S_misreport
                        # low p-value -> reject null -> there is envyfreeness violation
                        pvalue[k, l], tstat[k, l] = two_sample_paired_sign_test(S1=S1, S2=S2,
                                                                                alternative='less')

        raw_values_reported[np.diag_indices(n_groups)] = raw_values_true[
            np.diag_indices(n_groups)]

        if return_type == 'dict':
            out = {
                'gaps': raw_values_reported - raw_values_true,
                'raw_values_personalized_reported': raw_values_reported,
                'raw_values_personalized_true': raw_values_true,
                'pvalue': pvalue,
                'tstat': tstat,
            }
            return out
        if metric_name != 'auc':
            gain = list(raw_values_reported.flatten() - raw_values_true.flatten())
        else:
            gain = list(raw_values_true.flatten() - raw_values_reported.flatten())
        df_dict = self.data_dict.copy()
        df_dict.update({'sample_type': sample_type,
                        'model': model,
                        'model_type': model_type,
                        'group_name_1': [rg for rg in group_names for _ in range(len(group_names))],
                        'n_1': [n_rg for n_rg in sample_size for _ in range(len(sample_size))],
                        'group_name_2': group_names * len(group_names),
                        'n_2': sample_size * len(sample_size),
                        'metric_name': metric_name,
                        'value': raw_values_true.flatten(),
                        'value_generic': float('nan'),
                        'gain': gain,
                        'pvalue': pvalue.flatten(),
                        'tstat': tstat.flatten()})
        return pd.DataFrame(df_dict)
