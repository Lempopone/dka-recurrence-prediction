import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks



### МУЛЬТИМОДАЛЬНОСТЬ (KDE)

def analyze_shape(series):
    kde = stats.gaussian_kde(series)

    xs = np.linspace(series.min(), series.max(), 500)
    ys = kde(xs)

    ys = ys / ys.max()

    peaks, _ = find_peaks(
        ys,
        prominence=0.05,
        distance=25
    )

    return len(peaks)

### ECDF
def compute_ecdf(series):
    x = np.sort(series)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

### ОСНОВНОЙ АНАЛИЗ
def analyze_distributions(df: pd.DataFrame, target='target'):
    results = {}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if target in numeric_cols:
        numeric_cols.remove(target)
    if target in categorical_cols:
        categorical_cols.remove(target)

    def clean_series(series):
        s = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) < 10 or s.std() == 0:
            return None
        return s

    ### ЧИСЛОВЫЕ
    for col in numeric_cols:
        series = clean_series(df[col])
        if series is None:
            continue

        ### выбросы
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers_mask = (series < lower) | (series > upper)
        outliers_count = outliers_mask.sum()
        outliers_ratio = outliers_mask.mean()

        ### мультимодальность
        n_peaks = analyze_shape(series)

        ### метрики
        metrics = {
            'count': series.count(),
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'skew': stats.skew(series),
            'kurtosis': stats.kurtosis(series, fisher=False),
            'iqr': iqr,
            'outliers_count': int(outliers_count),
            'outliers_ratio': round(outliers_ratio, 4),
            'n_peaks': n_peaks
        }

        results[col] = metrics


        ### ВИЗУАЛИЗАЦИЯ

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3)

        ax1 = fig.add_subplot(gs[0:2, 0])  # hist
        ax2 = fig.add_subplot(gs[0, 1])    # boxplot
        ax3 = fig.add_subplot(gs[1, 1])    # violin
        ax4 = fig.add_subplot(gs[0:2, 2])  # ECDF
        ax5 = fig.add_subplot(gs[2, 0])    # log hist
        ax6 = fig.add_subplot(gs[2, 1:])   # text

        ### ГИСТОГРАММА
        sns.histplot(series, kde=True, ax=ax1)
        ax1.set_title(f'{col} — histogram + KDE')

        ### BOXPLOT
        temp = df[[col, target]].dropna()
        if not temp.empty:
            sns.boxplot(x=temp[target], y=temp[col], ax=ax2)
            ax2.set_title('boxplot vs target')

            sns.violinplot(x=temp[target], y=temp[col], ax=ax3)
            ax3.set_title('violin vs target')

        ### ECDF
        x_ecdf, y_ecdf = compute_ecdf(series)
        ax4.plot(x_ecdf, y_ecdf)
        ax4.set_title('ECDF')

        ### LOG SCALE
        if (series > 0).all():
            sns.histplot(np.log(series), kde=True, ax=ax5)
            ax5.set_title('log(x)')
        else:
            ax5.set_title('log not applicable')
            ax5.axis('off')

        ### TEXT
        ax6.axis('off')

        text = '\n'.join([
            f"count: {metrics['count']}",
            f"mean: {metrics['mean']:.4f}",
            f"median: {metrics['median']:.4f}",
            f"std: {metrics['std']:.4f}",
            f"min: {metrics['min']:.4f}",
            f"max: {metrics['max']:.4f}",
            "",
            f"skew: {metrics['skew']:.4f}",
            f"kurtosis: {metrics['kurtosis']:.4f}",
            f"IQR: {metrics['iqr']:.4f}",
            "",
            f"outliers: {metrics['outliers_count']} ({metrics['outliers_ratio']})",
            f"KDE peaks: {metrics['n_peaks']}"
        ])

        ax6.text(0, 1, text, va='top')

        plt.tight_layout()
        plt.show()

    ### КАТЕГОРИАЛЬНЫЕ
    for col in categorical_cols:
        series = df[col]
        proportions = series.value_counts(normalize=True, dropna=False)

        results[col] = proportions.to_dict()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        proportions.plot.pie(autopct='%1.1f%%', ax=axes[0])
        axes[0].set_title(col)
        axes[0].set_ylabel('')

        temp = df[[col, target]].copy()
        cross = pd.crosstab(temp[col], temp[target], normalize='index')
        cross.plot(kind='bar', stacked=True, ax=axes[1])

        plt.tight_layout()
        plt.show()

    return results



def detect_anomaly(df, threshold=10):
    """
    Поиск аномалий.

    Параметры:
    ----------
    df : pandas.DataFrame
        Измененный датафрейм.
    threshold : int, optional
        некоторый коэффициент для поиска аномалий

    Возвращает:
    ----------
    pandas.DataFrame
        измененный датафрейм.

    Описание:
    ----------
    Функция:
    - для каждой колонки проверяет наличие аномалии
    - если есть, удаляет
    """

    df_anomaly= df.copy()
    num_df = df_anomaly.select_dtypes(include=[np.number])
    rows_to_drop = set()

    for col in num_df.columns:
        q1 = num_df[col].quantile(0.25)
        q2 = num_df[col].quantile(0.5)
        q3 = num_df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5*iqr
        upper = q3 + 1.5*iqr

        mask = (num_df[col] > upper + threshold * iqr) | (num_df[col] < lower - threshold * iqr)
        anomalies  = num_df[col][mask]

        for idx, val in anomalies.items():
            if pd.isna(val):
                continue

            print(
                f"удалена аномалия в колонке: {col} \n"
                f"значение аномалии: {val}\n"
                f"превышает \n"
                f"- медиану в {val/q2:.2f} \n"
                f"- верхний ус в {val/upper:.2f}"
            )
            rows_to_drop.add(idx)

    df_anomaly.drop(index = rows_to_drop, inplace = True)

    print(f"было строк: {len(df)}")
    print(f"стало строк: {len(df_anomaly)}")

    return df_anomaly
