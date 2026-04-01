
import numpy as np
import scipy.stats as stats


def decide_transformations(results):
    transformations = {}

    for col, info in results.items():
        if 'skew' not in info:
            continue

        skew = info['skew']
        multimodal = info['multimodal']
        has_outliers = info['outliers_ratio'] > 0.03

        op = None

        if multimodal:
            transformations[col] = None
            continue

        if skew > 1:
            if info['min'] > 0:
                op = 'log'
            elif info['min'] >= 0:
                op = 'log1p'   # ← ВОТ ОН
            else:
                op = 'yeo_johnson'

        elif skew < -1:
            op = 'reflect_log'

        elif abs(skew) > 0.5 and has_outliers:
            op = 'robust'

        transformations[col] = op

    return transformations


def apply_transformations(df, transformations):
    df_trans = df.copy()

    for col, op in transformations.items():
        if op is None or col not in df_trans.columns:
            continue

        x = df_trans[col]

        if op == 'log':
            df_trans[col] = np.log(x)

        elif op == 'log1p':
            df_trans[col] = np.log1p(x)

        elif op == 'yeo_johnson':
            df_trans[col], _ = stats.yeojohnson(x)

        elif op == 'reflect_log':
            x_reflect = x.max() - x + 1
            df_trans[col] = np.log(x_reflect)

        elif op == 'robust':
            median = x.median()
            iqr = x.quantile(0.75) - x.quantile(0.25)
            df_trans[col] = (x - median) / (iqr + 1e-6)

    return df_trans