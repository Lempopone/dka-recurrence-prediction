# Preprocessing, Correlation Analysis, Visualization and Transformation Modules

## 1. Preprocessing Functions

### `change_cols_names()`
- Переименовывание русскоязычных названий колонок на английские

---

### `del_cols()`
- Удаляет:
  - колонки с ≥ 60% пропусков  
  - дубликаты строк  
  - строки с пропусками таргета  
  - квазиконстантные признаки  
  - мусорные колонки по заданному объекту `list`  
- Переносит "историю болезни" в индекс  

---

### `set_dtype()`
- Определяет и устанавливает нужную типизацию:
  - `object`
  - `category`
  - `float`
  - `int`
- Корректирует типы при необходимости  

---

### `basic_info()`
- Выводит таблицу:
  - кол-во пропусков  
  - % пропусков  
  - кол-во уникальных значений  
  - тип данных  
  - размер датасета (до / после)  

---

## 2. Correlation Functions

### `get_corr_duo()`
- Перебирает матрицу корреляций по Пирсону  
- Отбирает топ пары  

---

### `get_corr_triad()`
- Поиск триад корреляций вида:  
  `a ≈ alpha * b + beta * c`  
- Полный перебор всех перестановок  

---

### `del_corr_duo()`
- Удаляет по порогу парные корреляции (колонку/колонки)  
- Выводит список удаленных колонок  

---

### `del_corr_triad()`
- По созданному датафрейму триадных корреляций:
  - отсекает триады по порогу  
  - считает p-value каждой из 3 колонок к таргету  
  - удаляет наиболее статистически-незначимую  
- Выводит список удаленных колонок  

---

## 3. Visualization and Anomaly Detection Functions

### `detect_anomaly()`
- Для каждой колонки проверяет наличие аномалий  
- При наличии — удаляет  

---

### `analyze_shape()`
- Определение мультимодальности распределения  
- Основано на пиках плотности распределения  

---

### `compute_ecdf(series)`
- Создание эмпирической функции распределения  

---

### `analyze_distributions()`

#### Для числовых колонок:
- Гистограмма распределения  
- Ящик с усами по таргету  
- Violin-график по таргету  
- ECDF  
- Логистическая гистограмма распределения  
- Статистические метрики:
  - медиана  
  - среднее  
  - дисперсия  
  - IQR и др.  

#### Для категориальных и бинарных:
- Баланс классов по секторам  
- Распределение классов по таргету  

- Сохранение атрибутов каждой колонки в словарь  

---

## 4. Distribution Transformation Functions

### `decide_transformations()`
- На основе:
  - `skew`  
  - `multimodal`  
  - `outliers_ratio`  

- Выбирает метод трансформации:
  - `log` (если `min > 0`)  
  - `log1p` (если `min ≥ 0`)  
  - `yeo_johnson` (если есть отрицательные значения)  
  - `reflect_log` (при `skew < -1`)  
  - `robust` (при `|skew| > 0.5` и выбросах > 3%)  
  - `None` (мультимодальные или нормальные распределения)  

---

### `apply_transformations()`
- Применяет выбранные трансформации к колонкам  
- Возвращает преобразованный датафрейм  

---

## 5. Modeling and Training Pipeline

### `get_models()`
- Возвращает набор моделей:
  - XGBoost (tree / linear)  
  - Logistic Regression  
  - LightGBM  
  - Random Forest  
  - Decision Tree  
  - CatBoost  

---

### `get_preprocessor(model_name, cat_cols, num_cols)`
- Формирует препроцессинг в зависимости от модели:

#### Для `catboost`:
- Без преобразований (`passthrough`)

#### Для `logreg`:
- StandardScaler для числовых  
- OneHotEncoder для категориальных  

#### Для остальных моделей:
- OneHotEncoder для категориальных  
- Числовые — без изменений  

---

### `get_optuna_params(trial, model_name)`
- Генерация гиперпараметров для моделей через Optuna  
- Учитывает дисбаланс классов (`scale_pos_weight`)  

#### Поддерживаемые модели:
- XGBoost (tree / linear)  
- Logistic Regression  
- LightGBM  
- Random Forest  
- Decision Tree  
- CatBoost  

---

## 6. Threshold Optimization

### `find_best_threshold(y_true, y_pred, min_recall=0.7)`
- Подбор оптимального порога классификации  

#### Логика:
- Перебор threshold ∈ [0.01, 0.99]  
- Ограничение: recall ≥ заданного уровня  
- Цель: максимизация precision  

#### Если условие не выполнено:
- Выбирается threshold с минимальным отклонением по recall  

---

## 7. Nested Cross-Validation Training

### `train_full_pipeline_nested_cv(...)`

Полный цикл обучения моделей с nested CV.

#### Этапы:

1. Разделение данных:
   - train / test (stratified)

2. Outer CV:
   - оценка модели  

3. Inner CV:
   - подбор гиперпараметров (Optuna)  

4. Обучение модели:
   - с учетом препроцессинга  

5. Расчет threshold:
   - с учетом медицинского ограничения (recall)  

6. Сохранение моделей:
   - по каждому outer fold  

---

#### Сохраняемые результаты:
- OOF предсказания  
- Предсказания на тесте  
- Метрики по фолдам  
- Лучшие параметры  
- Thresholds  

---

## 8. Model Evaluation

### `summarize_results(results, y_train, y_test)`
- Вывод агрегированных метрик:

#### Метрики:
- ROC-AUC (train / val / test)  
- Стандартное отклонение  

#### Дополнительно:
- Медианный threshold  
- Classification report:
  - train (OOF)  
  - test  

---

## 9. Stacking

### `train_stacking_meta_model(results, model_names, y_train, y_test)`
- Обучение мета-модели (Logistic Regression)

#### Использует:
- OOF предсказания как признаки (train)  
- Test predictions как признаки (test)  

#### Выход:
- AUC на train и test  
- Обученная meta-модель  

---

### `search_best_stacking_combinations(results, y_train, y_test)`
- Перебор всех комбинаций моделей  

#### Логика:
- Перебор ансамблей размерности от 2 до N  
- Обучение meta-модели  
- Оценка по ROC-AUC  

#### Результат:
- Отсортированный список комбинаций по качеству  

---

## 10. SHAP Analysis

### `run_catboost_shap_ensemble(model_dir, X_test)`
- Интерпретация ансамбля CatBoost моделей  

#### Логика:
1. Загрузка моделей (5 фолдов)  
2. Расчет SHAP значений для каждой модели  
3. Усреднение SHAP  

---

#### Визуализация:
- Summary plot (dot)  
- Feature importance (bar)  

---

## 11. Key Characteristics of Training Pipeline

- Nested cross-validation (устойчивость оценки)  
- Optuna hyperparameter tuning  
- Балансировка классов  
- Оптимизация threshold под recall  
- Сохранение моделей по фолдам  
- Поддержка ансамблей и стекинга  
- Интерпретируемость через SHAP  

---