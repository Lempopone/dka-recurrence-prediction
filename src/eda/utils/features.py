import pandas as pd
import numpy as np


def add_missing_flags(df, threshold=0.25):
    df_flags = df.copy()
    n = len(df_flags)

    for col in df_flags.columns:
        missing_ratio = df_flags[col].isna().mean()

        if missing_ratio >= threshold:
            df_flags[col + "_isna"] = df_flags[col].isna().astype(int)

    return df_flags

def fill_cat(df, target = 'target'):
    df_cat=df.copy()
    cat_cols = df_cat.select_dtypes(include=['object', 'category', 'bool']).columns
    if target in cat_cols:
        cat_cols = cat_cols.drop(target)

    for col in cat_cols:
        if df_cat[col].isna().sum() == 0:
            continue

        df_cat[col] = df_cat[col].fillna(df_cat[col].mode().iloc[0])

    return df_cat

def fill_numeric(df, target='target'):
    df_num = df.copy()
    num_cols = df_num.select_dtypes(include=['int64', 'float64', 'Int64']).columns

    if target in num_cols:
        num_cols = num_cols.drop(target)

    corr_matrix = df_num[num_cols].corr()

    for col in num_cols:
        if df_num[col].isna().sum() == 0:
            continue

        # ищем лучший признак
        corr = corr_matrix[col].abs().sort_values(ascending=False)
        best = next((c for c in corr.index if c != col), None)

        if best is not None:
            try:
                # бинируем признак
                bins = pd.qcut(df_num[best], q=5, duplicates='drop')

                df_num[col] = df_num.groupby(bins, observed=True)[col].transform(
                    lambda x: x.fillna(x.median())
                )
            except:
                pass

        # fallback
        if pd.api.types.is_integer_dtype(df_num[col]):
            df_num[col] = df_num[col].fillna(round(df_num[col].median()))
        else:
            df_num[col] = df_num[col].fillna(df_num[col].median())

    return df_num