from postprocessing.postprocessing_abstract import *
from metrics.metric_classification import *


class AddRankingColumns(PostProcessingAbstract):
    optional_voting_fields = []
    common_voting_fields = ['size', 'p', 'n']

    voting_ensembles = {'sensitivity': [Sensitivity.name],
                        'specificity': [Specificity.name],
                        'balanced': [Phi.name]}

    CI_columns = [Sensitivity.name, Specificity.name]

    descending_column_names = ['size', 'p', 'n'] +  [f'{ci_metric}_CI_diff' for ci_metric in CI_columns]

    extended_voting_ensembles = {}
    for base_field in voting_ensembles.keys():
        extended_voting_ensembles[base_field] = voting_ensembles[base_field].copy()
        for f in optional_voting_fields:
            extended_voting_ensembles[f'{base_field}_with_{f}'] = voting_ensembles[base_field] + [f]
    voting_ensembles = extended_voting_ensembles

    for base_field in voting_ensembles:
        voting_ensembles[base_field] = voting_ensembles[base_field] + common_voting_fields

    @classmethod
    def process(cls, sfp, postprocessing_settings: dict):
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
        for ci_metric in cls.CI_columns:
            df[f'{ci_metric}_CI_diff_impact'] = (df[f'{ci_metric}_CI_diff'] > 0).astype(int)
        return df

    @staticmethod
    def preferImpactfulSlices(row, ci_metric, max_rank):
        rank = row[f'sort_on_{ci_metric}']
        diff_CI = row[f'{ci_metric}_CI_diff_impact']
        if diff_CI == 0:
            return None
        else:
            return rank
