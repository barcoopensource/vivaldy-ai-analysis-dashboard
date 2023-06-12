from postprocessing.postprocessing_abstract import *
from metrics.metric_classification import *


class AddRankingColumns(PostProcessingAbstract):

    @classmethod
    def process(cls, sfp, postprocessing_settings: dict):
        cls.optional_voting_fields = postprocessing_settings.get('optional_voting_fields', [])
        cls.common_voting_fields = postprocessing_settings.get('common_voting_fields', [])
        cls.voting_ensembles = postprocessing_settings.get('voting_ensembles', [])
        cls.CI_columns = postprocessing_settings.get('CI_columns', [])
        cls.extended_voting_ensembles = postprocessing_settings.get('extended_voting_ensembles', {})

        cls.descending_column_names = ['size', 'p', 'n'] +  [f'{ci_metric}_CI_diff' for ci_metric in cls.CI_columns]

        for base_field in cls.voting_ensembles.keys():
            cls.extended_voting_ensembles[base_field] = cls.voting_ensembles[base_field].copy()
            for f in cls.optional_voting_fields:
                cls.extended_voting_ensembles[f'{base_field}_with_{f}'] = cls.voting_ensembles[base_field] + [f]
        cls.voting_ensembles = cls.extended_voting_ensembles

        for base_field in cls.voting_ensembles:
            cls.voting_ensembles[base_field] = cls.voting_ensembles[base_field] + cls.common_voting_fields

        for d in sfp.degrees:
            df = sfp.dfs[d]
            df = cls.prepare_dataframe(df)
            for v in cls.voting_ensembles:
                aggregated_vote = 0
                for f in cls.voting_ensembles[v]:
                    ascending = not (f in cls.descending_column_names)
                    df_sorted = df[f].sort_values(ascending=ascending)
                    idx = df_sorted.index
                    vote = df_sorted.reset_index(drop=True).index.to_numpy()
                    df_sorted = pd.Series(data=vote, index=idx)
                    aggregated_vote += df_sorted
                df[f'sort_on_{v}'] = aggregated_vote

            # move the slices with positive CI_diff to the top of the list, but with the same order as found in aggregated_vote
            for ci_metric in cls.CI_columns:
                df[f'sort_on_{ci_metric}_impact'] = df.apply(lambda row: cls.preferImpactfulSlices(row, ci_metric, df[f'sort_on_{ci_metric}'].max()), axis=1)

            df.to_csv(sfp.data_output_dir / f'slices_degree{d}_sorted.csv')
            sfp.dfs[d] = df

    @classmethod
    def prepare_dataframe(cls, df):
        CI_columns_not_found = []
        for ci_metric in cls.CI_columns:
            ci_diff_column_name = f'{ci_metric}_CI_diff'
            if ci_diff_column_name in df.columns:
                df[f'{ci_diff_column_name}_impact'] = (df[ci_diff_column_name] > 0).astype(int)
            else:
                CI_columns_not_found.append(ci_metric)
        for ci_metric in CI_columns_not_found:
            cls.CI_columns.remove(ci_metric)

        # Check if all metrics in voting ensembles are found
        invalid_voting_ensembles = []
        for v in cls.voting_ensembles:
            for metric in cls.voting_ensembles[v]:
                if metric not in df.columns:
                    invalid_voting_ensembles.append(v)
                    break
        for v in invalid_voting_ensembles:
            cls.voting_ensembles.pop(v, None)
        return df

    @staticmethod
    def preferImpactfulSlices(row, ci_metric, max_rank):
        rank = row[f'sort_on_{ci_metric}']
        diff_CI = row[f'{ci_metric}_CI_diff_impact']
        if diff_CI == 0:
            return None
        else:
            return rank
