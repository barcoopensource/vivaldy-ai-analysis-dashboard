from postprocessing.postprocessing_abstract import *


class AddCIDifference(PostProcessingAbstract):
    @classmethod
    def process(cls, sfp, postprocessing_settings: dict):
        dfs = []
        for d in sfp.degrees:
            df = sfp.dfs[d]
            df['degree'] = d
            dfs.append(df)

        df_full = pd.concat(dfs)

        def __extract_reference_values(row, fields_parent_degree, parent_degree, metric_column_name, column_suffix=''):
            import numpy as np
            sel = df_full['degree'] == parent_degree
            if fields_parent_degree is not None:
                for f in fields_parent_degree:
                    shard_description_value = row[f]
                    sel &= df_full[f] == shard_description_value
            if not any(sel):
                return np.nan

            column_name = f'{metric_column_name}{column_suffix}'
            if column_name in df_full.columns:
                reference_value = df_full.loc[sel, column_name].iloc[0]
            else:
                reference_value = np.nan
            return reference_value

        def _add_CI_difference(row, metric):
            import numpy as np
            from itertools import combinations
            d = row['degree']
            if d == 0:
                return np.nan

            column_values = [s.split(':')[0] for s in row['description'].split('  ')]
            shard_descriptions = column_values

            if d == 1:
                df_degree_zero = df_full[df_full['degree'] == 0]
                if len(df_degree_zero) == 0:
                    return np.nan
                else:
                    d0 = df_degree_zero.iloc[0]
                    return d0[f'{metric}_LB'] - row[f'{metric}_UB']

            if pd.isna(row[f'{metric}_UB']) or ((row[f'{metric}_UB'] - row[f'{metric}_LB']) == 0.0):
                return np.nan
            impact = np.inf

            for shard_description in combinations(shard_descriptions, d - 1):
                reference_option_LB = __extract_reference_values(row, shard_description, d - 1, metric, column_suffix='_LB')
                shard_impact = reference_option_LB - row[f'{metric}_UB']
                if shard_impact < impact:
                    impact = shard_impact
            impact /= (row[f'{metric}_UB'] - row[f'{metric}_LB'])
            if impact == np.inf:
                impact = np.nan

            return impact

        CI_metrics = [c for c in df_full.columns if ((f'{c}_LB' in df_full.columns) and (f'{c}_UB' in df_full.columns))]
        for metric in CI_metrics:
            print('Adding CI diff for metric: ' + metric)
            df_full[f'{metric}_CI_diff'] = df_full.apply(lambda row: _add_CI_difference(row, metric), axis=1)

        for d in df_full['degree'].unique():
            df = df_full[df_full['degree'] == d]
            df.to_csv(sfp.data_output_dir / f'slices_degree{d}_with_CI_diff.csv')
            sfp.dfs[d] = df
