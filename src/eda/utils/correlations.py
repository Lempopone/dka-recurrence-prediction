import numpy as np
from itertools import combinations
import pandas as pd
from scipy.stats import pointbiserialr

def get_corr_duo(df, n=5):
    """
    Поиск попарных корреляций.

    Параметры:
    ----------
    df : pandas.DataFrame
        Измененный датафрейм.
    n : int, optional
        Количество пар

    Возвращает:
    ----------
    Series.series
        Топ N пар корреляций.

    Описание:
    ----------
    Функция:
    - перебирает матрицу корреляций
    - отбирает топ пары
    """
    df_corr = df.copy()
    pairs_to_drop = set()
    numeric_cols = df_corr.select_dtypes(include=[np.number]).columns

    for i in range(0, df_corr[numeric_cols].shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((numeric_cols[i], numeric_cols[j]))

    au_corr = df_corr[numeric_cols].corr().abs().unstack()
    au_corr = au_corr.drop(labels=pairs_to_drop).sort_values(ascending=False)
    print("Top Absolute Correlations")
    return au_corr[0:n]

def get_corr_triad(
    df,
    threshold=0.85,
    target_name='target',
    alpha=1.0,
    beta=1.0
):
    """
    Поиск триад корреляций вида:
        a ≈ alpha*b + beta*c
    с полным перебором всех перестановок.

    Параметры:
    ----------
    df : pandas.DataFrame
    threshold : float
        Порог корреляции (фильтрация внутри функции)
    target_name : str
        Название таргета (исключается)
    alpha, beta : float
        Коэффициенты линейной комбинации

    Возвращает:
    ----------
    pandas.DataFrame с колонками:
        a, b, c, corr
    отсортированный по убыванию |corr|
    """

    df_copy = df.copy()

    # только числовые признаки
    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns.tolist()

    # убираем таргет
    if target_name in numeric_cols:
        numeric_cols.remove(target_name)

    results = []

    # перебор уникальных троек
    for col1, col2, col3 in combinations(numeric_cols, 3):

        cols = [col1, col2, col3]

        # перебор всех перестановок: кто будет "a"
        for i in range(3):
            a = cols[i]
            b = cols[(i + 1) % 3]
            c = cols[(i + 2) % 3]

            sub = df_copy[[a, b, c]].dropna()

            if sub.shape[0] < 2:
                continue

            try:
                combo = alpha * sub[b] + beta * sub[c]

                eps = 1e-12
                if sub[a].std() < eps or combo.std() < eps:
                    continue

                # проверка на нулевую дисперсию
                if sub[a].std() == 0 or combo.std() == 0:
                    continue

                corr = sub[a].corr(combo)

                if pd.notna(corr) and abs(corr) >= threshold:
                    results.append((a, b, c, corr))

            except Exception:
                continue

    df_res = pd.DataFrame(results, columns=['a', 'b', 'c', 'corr'])

    if not df_res.empty:
        df_res = df_res.sort_values(by='corr', key=lambda x: x.abs(), ascending=False).reset_index(drop=True)

    return df_res

def del_corr_triad(df, threshold=0.85, target_name='target'):

    """
    Удаление корреляций по типу a=b+c.

    Параметры:
    ----------
    df : pandas.DataFrame
        Измененный датафрейм.
    threshold : float, optional
        Порог отсечения корреляции
    target_name : str, optional
        Название таргетной переменной.

    Возвращает:
    ----------
    pandas.DataFrame
        исправленный датафрейм.

    Описание:
    ----------
    Функция:
    - По созданному датафрейму триадных корреляций отсекает триады по порогу
    - считает p-value каждой из 3 колонок к таргету
    - удаляет наиболее статистически-незначимую
    - выводит список удаленных колонок
    """
    df_triad = df.copy()
    df_drop = set()

    while True:

        df_del = get_corr_triad(df_triad, threshold=threshold)
        num_cols = df_del.shape[1]

        filtered = df_del[df_del['corr'].abs() > threshold]

        # если больше нет сильных триад → выходим
        if filtered.empty:
            break

        del_cols = filtered.iloc[0, 0:(num_cols-1)].tolist()

        results = []

        for col in del_cols:
            sub = df_triad[[col, target_name]].dropna()

            if sub.empty:
                continue

            X = sub[col]
            y = sub[target_name].astype(int)

            try:
                corr, pval = pointbiserialr(y, X)
                results.append((col, pval))
            except:
                continue

        # ищем max p-value
        max_pval = -1
        drop_col = None

        for col, pval in results:
            if pval > max_pval:
                max_pval = pval
                drop_col = col

        if drop_col is None:
            break

        print(f" применена функция del_triad_correlations, удалены колонки: \n{drop_col}, p-value = {max_pval:.5f}")

        df_drop.add(drop_col)
        df_triad = df_triad.drop(columns=[drop_col], errors='ignore')

    return df_triad

def del_corr_duo(df, threshold=0.9):
    """
    Удаление попарных корреляций.

    Параметры:
    ----------
    df : pandas.DataFrame
        Измененный датафрейм.
    threshold : float, optional
        Порог отсечения корреляции

    Возвращает:
    ----------
    pandas.DataFrame
        исправленный датафрейм.

    Описание:
    ----------
    Функция:
    - Удаляет по порогу парные корреляции (колонку/колонки),
    - выводит список удаленных колонок
    """

    df_duo = df.copy()
    col_corr = set() # Set of all the names of deleted columns
    top_corr = get_corr_duo(df, 50)
    name = 1
    for (col1,col2), corr_value in top_corr.items():
            if (corr_value >= threshold) and (name not in col_corr):
                name = col1
                col_corr.add(name)
                df_duo.drop(name, axis=1, inplace=True)
    print(f'применена функция del_corr_duo, удалены колонки: {col_corr}')
    return df_duo