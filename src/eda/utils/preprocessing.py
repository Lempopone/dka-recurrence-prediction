import pandas as pd
import numpy as np

def change_cols_names(df):
    change_cols_df = df.copy()
    new_columns = [
        'row_id',
        'birth_date',
        'medical_record_id',
        'diabetes_type',
        'age',
        'sex',
        'dka_date',
        'diabetes_duration',
        'diabetes_onset_age',
        'insulin_therapy_type',
        'continuous_glucose_monitoring_system',
        'daily_insulin_dosage',
        'severe_hypoglycemia_history',
        'mild_hypoglycemia_per_week',
        'dka_history_count',
        'ckd_stage_c',
        'ckd_stage_a',
        'neuropathy',
        'retinopathy_stage',
        'target_hba1c',
        'hba1c',
        'creatinine_admission',
        'urea_admission',
        'ph_admission',
        'be_admission',
        'lactate_admission',
        'potassium_admission',
        'sodium_admission',
        'glucose_admission',
        'cholesterol_total',
        'ldl',
        'hdl',
        'triglycerides',
        'ph_normalization_days',
        'dka_severity',
        'alcohol_before_dka',
        'substance_use_before_dka',
        'beck_depression_score',
        'hypoglycemia_fear_score',
        'cv_glucose',
        'time_below_range_4',
        'gmi',
        'tar1',
        'tar2',
        'tir',
        'tbr1',
        'tbr2',
        'death_flag',
        'target'
    ]

    change_cols_df.columns = new_columns

    return change_cols_df

def del_cols(
        df,
        target_name = 'target',
        threshold = 0.6,
        medical_history = 'medical_record_id',
        waste_cols = None
):
    """
    Очистка датафрейма от нерелевантных данных.

    Параметры:
    ----------
    df : pandas.DataFrame
        Исходный датафрейм.
    target_name : str, optional
        Название таргетной переменной.
    threshold : float, optional
        Порог доли пропусков (0–1) для удаления колонок.
    medical_history : str, optional
        Колонка с историей болезни (будет установлена как индекс).
    waste_cols : list of str, optional
        Список колонок для удаления по экспертному решению.

    Возвращает:
    ----------
    pandas.DataFrame
        Очищенный датафрейм.

    Описание:
    ----------
    Функция:
    - удаляет строки с пропусками в таргете
    - удаляет дубликаты
    - удаляет колонки с долей пропусков выше threshold
    - удаляет заданные "мусорные" колонки
    - устанавливает индекс по medical_history
    - удаляет квазиконстантные признаки
    """

    if waste_cols is None:
        waste_cols = ['row_id', 'target_hba1c', 'dka_date', 'alcohol_before_dka']

    del_df = df.copy()

    ### удаляем пропуски в таргетах
    del_df = del_df.dropna(subset=[target_name])

    ### удаляем дубликаты
    del_df = del_df.drop_duplicates().reset_index(drop=True)

    ### удаляем колонки с более 60% пропусков
    missing_values = del_df.isnull().sum()
    columns_to_drop = missing_values[missing_values>=(del_df.shape[0]*threshold)].index.tolist()
    columns_to_drop.extend(waste_cols) ### ненужная колонка с номером пациента и целевым HbA1c
    del_df.drop(columns = columns_to_drop,  inplace=True)

    ### ставим medical_history в индекс
    del_df.set_index(medical_history, inplace=True)
    del_df.index = del_df.index.astype(int)

    ### удаляем колонки квазиконстантные
    const_cols = []
    for col in del_df.columns:
        if del_df[col].nunique() <= 5:
            value_counts = del_df[col].value_counts()
            max = value_counts.max() ### кол-во значений макс класса
            sum_others = value_counts.sum() - max ### кол-во значений остальных
            if sum_others/max <=0.1:
                const_cols.append(col)

    del_df.drop(columns = const_cols, inplace=True)
    columns_to_drop.extend(const_cols)

    print(f"После выполнения функции del_cols удалены колонки: \n{columns_to_drop}")
    print(f"Из них константные: \n{const_cols}")


    return del_df


def set_dtype(df):
    """
    Замена типизации колонок

    Параметры:
    ----------
    df : pandas.DataFrame
        Исправленный датафрейм.

    Возвращает:
    ----------
    pandas.DataFrame
        Измененный датафрейм.

    Описание:
    ----------
    Функция:
    - меняет типизацию колонок
    - datetime в года int
    - с 2 уникальными значениями в boolean
    - оценивает, насколько колонка числовая

    """
    df_dtype = df.copy()
    base_types = df.dtypes.copy()  # типы до изменений
    for col in df_dtype.columns:

        if df_dtype[col].dtype == 'datetime64[ns]':
            df_dtype[col] = (2026 - df_dtype[col].dt.year).astype('Int64')

        elif pd.api.types.is_float_dtype(df_dtype[col]) and df_dtype[col].nunique(dropna=True) > 8:
            non_na = df_dtype[col].dropna()

            if len(non_na) > 0:
                # проверка "почти все целые"
                if ((non_na % 1) == 0).mean() > 0.9:
                    df_dtype[col] = df_dtype[col].round().astype('Int64')


        elif df_dtype[col].nunique(dropna=True) == 2:
            df_dtype[col] = df_dtype[col].astype('Int64')

        elif 2 < df_dtype[col].nunique(dropna=True) <= 8:
            df_dtype[col] = pd.Categorical(
                pd.Series(pd.factorize(df_dtype[col])[0]).replace(-1, pd.NA)
            )

        elif df_dtype[col].dtype == 'object' and df_dtype[col].nunique(dropna=True) > 8:
            cleaned = df_dtype[col].astype(str) \
                .str.replace(',', '.', regex=False) \
                .str.replace(r'[^0-9.\-]', '', regex=True)

            numeric = pd.to_numeric(cleaned, errors='coerce')
            # проверка "насколько это число"
            mask = df_dtype[col].notna()
            if numeric[mask].notna().mean() > 0.8:
                is_int = (numeric.dropna() % 1 == 0).mean() > 0.8

                if is_int:
                     df_dtype[col] = numeric.round().astype('Int64')
                else:
                    df_dtype[col] = numeric.astype('float64')

    new_types = df_dtype.dtypes.copy()

    compare = pd.concat(
        [base_types, new_types],
        axis=1
    )
    compare.columns = ['Было', 'Стало']
    compare.index.name = 'Колонка'

    print(compare)

    return df_dtype

def basic_info(df, base_df):
    ### делаем копию
    """
    Вывод характеристик колонок.

    Параметры:
    ----------
    df : pandas.DataFrame
        Измененный датафрейм.
    base_df : pandas.DataFrame
        Исходный датафрейм.

        Возвращает:
    ----------
    pandas.DataFrame
        датафрейм со статистикой о колонках.

    Описание:
    ----------
    Функция:
    - Считает кол-во пропущенных значений
    - их % от всего числа строк для каждой колонки
    - выводит тип колонки и кол-во уникальных значений
    """

    info_df = df.copy()

    print(f"""
      строки:
        базовый df: {base_df.shape[0]}
        исправленный df: {info_df.shape[0]}
      колонки:
        базовый df: {base_df.shape[1]}
        исправленный df: {info_df.shape[1]}
    """)

    ### выводим количество пропущенных значений+процент от общего числа строк+тип колонки
    missing_count = info_df.isnull().sum()
    missing_percent = (100*missing_count/info_df.shape[0]).round(2)
    cols_types = info_df[missing_count.index.tolist()].dtypes
    unique_count = info_df.nunique()

    missing_df = pd.concat(
        [missing_count, missing_percent, unique_count, cols_types],
        axis=1,
        keys = ['Кол-во пропусков', 'Процент пропусков %', 'Кол-во уник. значений', 'Тип колонки']
    )
    missing_df.sort_values("Кол-во пропусков", ascending=False, inplace=True)

    return missing_df
