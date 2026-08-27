# German Credit Risk — XGBoost Credit Risk Classification

Проект по прогнозированию кредитного риска клиентов на основе классических алгоритмов машинного обучения.

Основная задача — предсказать, является ли клиент кредитным риском, а затем проанализировать не только качество модели, но и причины её решений с помощью SHAP.

Финальная модель проекта — **XGBoost** с оптимизированным порогом классификации `0.236`.

---

## 1. Цель проекта

Цель — построить модель бинарной классификации для прогнозирования кредитного риска.

Целевая переменная:

- `Risk = 0` — хороший кредитный риск;
- `Risk = 1` — плохой кредитный риск.

Особое внимание уделяется обнаружению клиентов класса `Risk = 1`, поэтому помимо Accuracy используются Precision, Recall, F1 и ROC-AUC.

---

## 2. Данные

Размер исходного датасета:

```text
1000 observations
11 original features
```

Распределение целевой переменной:

| Risk | Количество | Доля |
|---|---:|---:|
| 0 | 700 | 70% |
| 1 | 300 | 30% |

Датасет является умеренно несбалансированным.

---

## 3. Общий pipeline

```text
Raw data
    ↓
Data loading
    ↓
Target preparation
    ↓
Feature engineering
    ↓
Train / Validation / Test split
    ↓
Preprocessing
    ↓
Baseline models
    ↓
Cross-validation
    ↓
XGBoost hyperparameter tuning
    ↓
Threshold tuning
    ↓
Final XGBoost
    ↓
Test evaluation
    ↓
Feature importance
    ↓
Permutation importance
    ↓
SHAP analysis
    ↓
TP / TN / FP / FN analysis
```

---

## 4. Разделение данных

Используется стратифицированное разделение:

```text
Train:      600
Validation: 200
Test:       200
```

Стратификация позволяет сохранить примерно одинаковое соотношение классов в каждом наборе.

Test set не используется для выбора модели, гиперпараметров или threshold.

---

## 5. Feature Engineering

Перед обучением модели выполняется feature engineering.

В проекте используются дополнительные признаки, среди которых:

- `Log Monthly payment`;
- `Payment_to_age`;
- `Age after loan`;
- `Young`.

Feature engineering выполняется до построения финального pipeline.

---

## 6. Preprocessing

Предобработка вынесена в отдельный модуль.

Pipeline включает:

- обработку числовых признаков;
- обработку категориальных признаков;
- ordinal encoding;
- one-hot encoding;
- необходимые преобразования признаков.

Все preprocessing steps входят в `sklearn Pipeline`, поэтому при обучении они подгоняются только на соответствующей обучающей выборке.

---

## 7. Baseline Models

В проекте сравниваются пять моделей:

1. Logistic Regression
2. Random Forest
3. LightGBM
4. XGBoost
5. CatBoost

### Validation results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.720 | 0.571 | 0.267 | 0.364 | 0.731 |
| Random Forest | 0.655 | 0.385 | 0.250 | 0.303 | 0.699 |
| LightGBM | 0.705 | 0.511 | 0.400 | 0.449 | 0.685 |
| XGBoost | 0.705 | 0.509 | 0.450 | 0.478 | 0.709 |
| CatBoost | 0.675 | 0.447 | 0.350 | 0.393 | 0.714 |

---

## 8. Cross-Validation

Для оценки устойчивости моделей используется **5-fold cross-validation**.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.695 ± 0.025 | 0.509 ± 0.107 | 0.250 ± 0.039 | 0.328 ± 0.031 | 0.707 ± 0.028 |
| Random Forest | 0.750 ± 0.028 | 0.647 ± 0.100 | 0.394 ± 0.037 | 0.487 ± 0.042 | 0.756 ± 0.054 |
| LightGBM | 0.737 ± 0.035 | 0.577 ± 0.076 | 0.461 ± 0.054 | 0.513 ± 0.063 | 0.740 ± 0.059 |
| XGBoost | 0.725 ± 0.038 | 0.557 ± 0.086 | 0.433 ± 0.072 | 0.485 ± 0.071 | 0.743 ± 0.054 |
| CatBoost | 0.743 ± 0.033 | 0.612 ± 0.093 | 0.417 ± 0.068 | 0.492 ± 0.067 | 0.746 ± 0.061 |

На этапе cross-validation лучшим по ROC-AUC среди baseline-моделей оказался Random Forest:

```text
ROC-AUC = 0.756 ± 0.054
```

Random Forest показал лучший средний CV ROC-AUC среди baseline-моделей.
XGBoost был выбран для дальнейшей оптимизации как модель, показавшая
конкурентное качество и потенциал для улучшения за счёт hyperparameter
tuning и оптимизации decision threshold.

---

## 9. XGBoost Hyperparameter Tuning

Для XGBoost используется `RandomizedSearchCV`.

Лучшие параметры:

```python
{
    "subsample": 0.7,
    "n_estimators": 200,
    "min_child_weight": 1,
    "max_depth": 3,
    "learning_rate": 0.01,
    "gamma": 0.5,
    "colsample_bytree": 0.8
}
```

Лучший CV ROC-AUC:

```text
0.769
```

---

## 10. Threshold Tuning

После выбора XGBoost оптимизируется не только модель, но и decision threshold.

Стандартный threshold:

```text
0.50
```

Оптимальный threshold по validation:

```text
0.236
```

Validation F1:

```text
0.565
```

Такое снижение threshold позволяет модели чаще относить клиентов к классу `Risk`.

Это повышает Recall, но одновременно увеличивает количество False Positives.

---

## 11. Финальная модель

После выбора:

- модели;
- гиперпараметров;
- decision threshold;

финальный XGBoost обучается на:

```text
Train + Validation = 800 observations
```

Test set остаётся полностью независимым до финальной оценки.

---

## 12. Финальные результаты на Test

| Метрика | Значение |
|---|---:|
| Accuracy | **0.640** |
| Precision | **0.446** |
| Recall | **0.833** |
| F1 | **0.581** |
| ROC-AUC | **0.792** |

Финальная конфигурация:

```text
Model: XGBoost
Threshold: 0.236
Best CV ROC-AUC: 0.769
Test ROC-AUC: 0.792
```

> ROC-AUC рассчитывается по вероятностям модели и не зависит от выбранного
> decision threshold. Threshold `0.236` влияет на Accuracy, Precision, Recall
> и F1, но не изменяет ROC-AUC.

---

## 13. Confusion Matrix

На тестовой выборке:

```text
              Predicted
              0     1
Actual 0     78    62
Actual 1     10    50
```

Таким образом:

```text
TN = 78
FP = 62
FN = 10
TP = 50
```

Особенно важно, что:

```text
Recall = TP / (TP + FN)
       = 50 / (50 + 10)
       = 0.833
```

То есть модель обнаруживает около **83.3% рискованных клиентов**.

Цена этого — относительно большое количество False Positives.

---

## 14. Анализ threshold

Финальный decision threshold `0.236` был выбран **только на validation set**
по максимальному F1 и затем зафиксирован до оценки на test set.

Дополнительно после финальной оценки была выполнена диагностическая
проверка нескольких thresholds на test set. Эта таблица используется
только для анализа trade-off между Precision, Recall, F1 и Accuracy
и не использовалась для выбора финального threshold.

| Threshold | Precision | Recall | F1 | Accuracy |
|---:|---:|---:|---:|---:|
| 0.20 | 0.415 | 0.900 | 0.568 | 0.590 |
| 0.25 | 0.476 | 0.817 | 0.601 | 0.675 |
| 0.30 | 0.500 | 0.750 | 0.600 | 0.700 |
| 0.35 | 0.541 | 0.667 | 0.597 | 0.730 |
| 0.40 | 0.596 | 0.567 | 0.581 | 0.755 |
| 0.45 | 0.646 | 0.517 | 0.574 | 0.770 |
| 0.50 | 0.806 | 0.417 | 0.549 | 0.795 |

---

## 15. Feature Importance

Встроенная XGBoost importance показывает следующие наиболее значимые признаки:

| Feature | Importance |
|---|---:|
| Checking account | 0.1747 |
| Duration | 0.0733 |
| Housing — rent | 0.0645 |
| Saving accounts | 0.0592 |
| Credit amount | 0.0549 |
| Monthly payment | 0.0542 |
| Housing — own | 0.0537 |
| Log Monthly payment | 0.0508 |
| Purpose — radio/TV | 0.0493 |
| Sex — female | 0.0446 |
| Payment to age | 0.0401 |
| Sex — male | 0.0379 |
| Age | 0.0351 |
| Housing — free | 0.0348 |
| Purpose — education | 0.0340 |

---

## 16. Permutation Importance

Permutation importance рассчитан на Test set с использованием ROC-AUC.

Наиболее важные признаки:

| Feature | Importance |
|---|---:|
| Checking account | 0.1516 |
| Credit amount | 0.0308 |
| Monthly payment | 0.0158 |
| Duration | 0.0129 |
| Age after loan | 0.0061 |
| Age | 0.0056 |
| Saving accounts | 0.0024 |
| Housing | 0.0013 |
| Purpose | 0.0008 |
| Young | 0.0003 |

Checking account остаётся наиболее важным признаком и при permutation importance.

---

# 17. SHAP Analysis

Для интерпретации XGBoost используется библиотека **SHAP**.

SHAP позволяет определить:

- какие признаки сильнее всего влияют на prediction;
- направление влияния признаков;
- индивидуальные причины prediction;
- особенности правильных и ошибочных решений.

### Global SHAP importance

| Feature | Mean \|SHAP\| |
|---|---:|
| Checking account | **0.4987** |
| Duration | **0.1809** |
| Credit amount | **0.1379** |
| Monthly payment | **0.1342** |
| Saving accounts | **0.1017** |
| Housing — own | 0.0531 |
| Age | 0.0322 |
| Log Monthly payment | 0.0299 |
| Age after loan | 0.0237 |
| Purpose — radio/TV | 0.0230 |


**Mean absolute SHAP показывает силу влияния признака независимо от направления.**

Для анализа направления влияния используется **signed SHAP**:
положительное значение увеличивает model output, отрицательное — уменьшает его.


Главные факторы модели:

```text
Checking account
Duration
Credit amount
Monthly payment
Saving accounts
```

---

# 18. SHAP Error Analysis

Дополнительно выполняется SHAP-анализ ошибок модели.

Каждое тестовое наблюдение относится к одной из четырёх групп:

- TP — True Positive;
- TN — True Negative;
- FP — False Positive;
- FN — False Negative.

Распределение:

| Group | Count | Описание |
|---|---:|---|
| TP | 50 | Risk правильно определён как Risk |
| TN | 78 | Good правильно определён как Good |
| FP | 62 | Good ошибочно определён как Risk |
| FN | 10 | Risk ошибочно определён как Good |

---

## 19. SHAP для False Positives

Количество:

```text
FP = 62
```

Наиболее важные признаки:

```text
Checking account
Duration
Monthly payment
Credit amount
Saving accounts
Housing — own
Age
Log Monthly payment
Sex — female
Age after loan
```

Mean absolute SHAP:

| Feature | FP |
|---|---:|
| Checking account | 0.3832 |
| Duration | 0.1457 |
| Monthly payment | 0.1374 |
| Credit amount | 0.1309 |
| Saving accounts | 0.1105 |
| Housing — own | 0.0596 |
| Age | 0.0308 |
| Log Monthly payment | 0.0292 |

FP — это клиенты с реальным `Risk = 0`, которых модель считает рискованными.

---

## 20. SHAP для False Negatives

Количество:

```text
FN = 10
```

Наиболее важные признаки:

```text
Checking account
Duration
Credit amount
Monthly payment
Saving accounts
Age
Housing — own
Age after loan
Payment to age
Log Monthly payment
```

Mean absolute SHAP:

| Feature | FN |
|---|---:|
| Checking account | 0.5305 |
| Duration | 0.2401 |
| Credit amount | 0.1572 |
| Monthly payment | 0.1351 |
| Saving accounts | 0.0804 |
| Age | 0.0547 |
| Housing — own | 0.0467 |
| Age after loan | 0.0309 |

FN — наиболее критичный тип ошибки для задачи обнаружения риска: это реальные рискованные клиенты, которых модель пропустила.

---

## 21. FP vs FN

Сравнение mean absolute SHAP:

| Feature | FP | FN | FN − FP |
|---|---:|---:|---:|
| Checking account | 0.3832 | 0.5305 | 0.1473 |
| Duration | 0.1457 | 0.2401 | 0.0944 |
| Saving accounts | 0.1105 | 0.0804 | -0.0301 |
| Credit amount | 0.1309 | 0.1572 | 0.0263 |
| Age | 0.0308 | 0.0547 | 0.0239 |
| Purpose — business | 0.0042 | 0.0228 | 0.0186 |
| Housing — own | 0.0596 | 0.0467 | -0.0128 |
| Age after loan | 0.0208 | 0.0309 | 0.0101 |
| Payment to age | 0.0173 | 0.0246 | 0.0073 |
| Housing — rent | 0.0097 | 0.0035 | -0.0063 |
| Log Monthly payment | 0.0292 | 0.0235 | -0.0057 |
| Monthly payment | 0.1374 | 0.1351 | -0.0023 |

Наиболее заметные различия:

```text
Checking account
Duration
Credit amount
Age
Purpose — business
```

---

## 22. Направление влияния

Signed SHAP показывает направление влияния признака на model output.

### FP

Наиболее заметное положительное влияние:

```text
Checking account
Duration
Credit amount
Age
Sex — female
```

Наиболее заметное отрицательное влияние:

```text
Monthly payment
Log Monthly payment
Payment to age
Housing — free
Purpose — business
```

### FN

Наиболее заметное положительное влияние:

```text
Age
Purpose — business
Age after loan
Sex — female
Purpose — radio/TV
```

Наиболее заметное отрицательное влияние:

```text
Checking account
Duration
Monthly payment
Housing — own
Credit amount
```

Это показывает, что у FN несколько сильных признаков могут двигать prediction в сторону класса `0`, несмотря на фактический `Risk = 1`.

---

## 23. Representative observations

Для каждой группы выбирается representative observation.

Текущие representative observations:

```text
TP: test index 2
TN: test index 10
FP: test index 0
FN: test index 22
```

Пример:

```text
FP:
true = 0
pred = 1
probability = 0.330
```

```text
FN:
true = 1
pred = 0
probability = 0.121
```

Для этих наблюдений строятся SHAP waterfall plots, позволяющие увидеть вклад отдельных признаков в конкретное решение модели.

---

## 24. SHAP Reports

Результаты SHAP error analysis сохраняются в:

```text
reports/shap_errors/
```

В директории находятся:

```text
TP_importance.csv
TN_importance.csv
FP_importance.csv
FN_importance.csv
```

а также графики:

```text
TP_bar.png
TN_bar.png
FP_bar.png
FN_bar.png

TP_beeswarm.png
TN_beeswarm.png
FP_beeswarm.png
FN_beeswarm.png

TP_waterfall.png
TN_waterfall.png
FP_waterfall.png
FN_waterfall.png
```

Также сохраняется:

```text
test_predictions_with_errors.csv
```

с prediction, probability и типом ошибки для каждого объекта.

---

# 25. Структура проекта

```text
German-Credit-Risk/
│
├── data/
│   └── ...
│
├── models/
│   └── xgboost_final.joblib
│
├── reports/
│   ├── test_data.csv
│   ├── test_predictions.csv
│   └── shap_errors/
│       ├── TP_importance.csv
│       ├── TN_importance.csv
│       ├── FP_importance.csv
│       ├── FN_importance.csv
│       ├── TP_bar.png
│       ├── TN_bar.png
│       ├── FP_bar.png
│       ├── FN_bar.png
│       ├── TP_beeswarm.png
│       ├── TN_beeswarm.png
│       ├── FP_beeswarm.png
│       ├── FN_beeswarm.png
│       ├── TP_waterfall.png
│       ├── TN_waterfall.png
│       ├── FP_waterfall.png
│       ├── FN_waterfall.png
│       └── test_predictions_with_errors.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── cv.py
│   ├── shap_analysis.py
│   └── train.py
│
├── requirements.txt
└── README.md
```

---

# 26. Установка

Создание virtual environment:

```bash
python -m venv .venv
```

Активация:

### macOS / Linux

```bash
source .venv/bin/activate
```

Установка зависимостей:

```bash
pip install -r requirements.txt
```

---

# 27. Запуск обучения

Основная программа:

```bash
python -m src.train
```

В процессе выполняются:

- загрузка данных;
- feature engineering;
- train/validation/test split;
- обучение baseline-моделей;
- cross-validation;
- XGBoost hyperparameter tuning;
- threshold tuning;
- обучение финального XGBoost;
- оценка на test;
- feature importance;
- permutation importance;
- сохранение финальной модели и результатов.

После выполнения:

```text
Final model saved to:
models/xgboost_final.joblib
```

---

# 28. Запуск SHAP analysis

SHAP-анализ вынесен в отдельную программу:

```bash
python -m src.shap_analysis
```

Программа:

1. загружает финальный XGBoost;
2. загружает test data;
3. получает predictions;
4. формирует TP/TN/FP/FN;
5. рассчитывает SHAP values;
6. считает global SHAP importance;
7. считает SHAP importance отдельно для TP/TN/FP/FN;
8. строит bar plots;
9. строит beeswarm plots;
10. выбирает representative observations;
11. строит waterfall plots;
12. сохраняет результаты в `reports/shap_errors/`.

---

# 29. Используемые технологии

Основной стек:

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
- SHAP
- Matplotlib

---

# 30. Основные метрики

### Accuracy

Доля правильных предсказаний:

```text
(TP + TN) / (TP + TN + FP + FN)
```

### Precision

Доля действительно рискованных клиентов среди предсказанных как Risk:

```text
TP / (TP + FP)
```

### Recall

Доля найденных рискованных клиентов:

```text
TP / (TP + FN)
```

### F1

Гармоническое среднее Precision и Recall.

### ROC-AUC

Оценка способности модели разделять два класса независимо от конкретного threshold.

---

# 31. Почему важен Recall

В credit risk задача False Negative может быть более критичной, чем False Positive.

```text
False Positive:
Good client → Risk
```

Модель ошибочно считает хорошего клиента рискованным.

```text
False Negative:
Risk client → Good
```

Модель пропускает реального рискованного клиента.

Поэтому в проекте threshold снижен до `0.236`, чтобы увеличить чувствительность к Risk-классу.

Итог:

```text
Recall = 0.833
```

при этом:

```text
Precision = 0.446
```

Это сознательный trade-off.

---

# 32. Основные выводы

### 1. Checking account — главный фактор

Checking account является наиболее важным признаком по SHAP:

```text
Mean |SHAP| = 0.4987
```

Он также занимает первое место по permutation importance.

### 2. Duration — второй важный фактор

Срок кредита существенно влияет на решение модели:

```text
Mean |SHAP| = 0.1809
```

### 3. Credit amount и Monthly payment

Размер кредита и ежемесячный платёж являются одними из основных финансовых факторов модели.

### 4. Ошибки имеют структуру

Ошибки не выглядят полностью случайными.

Наиболее важными признаками для FP и FN остаются:

```text
Checking account
Duration
Credit amount
Monthly payment
Saving accounts
```

### 5. FN имеют особенно сильный вклад Checking account и Duration

Для FN:

```text
Checking account = 0.5305
Duration = 0.2401
```

против:

```text
FP Checking account = 0.3832
FP Duration = 0.1457
```

Это может быть полезным направлением для дальнейшего анализа пропущенных рискованных клиентов.

---

# 33. Ограничения проекта

Проект является учебным/портфолио-проектом и не предназначен для непосредственного принятия реальных кредитных решений.

Основные ограничения:

- небольшой датасет — 1,000 наблюдений;
- test set содержит только 200 наблюдений;
- классы несбалансированы;
- threshold оптимизирован по F1;
- не выполнялась probability calibration;
- SHAP объясняет поведение модели, но не доказывает причинно-следственные связи;
- не выполнялся полноценный fairness analysis;
- отсутствуют production monitoring и drift detection.

Для production credit-risk системы потребовались бы дополнительные проверки стабильности, calibration, fairness, data drift, model governance и regulatory validation.

---

# 34. Возможные улучшения

Следующие шаги:

1. Оптимизация threshold с учётом реальной стоимости FP и FN.
2. Probability calibration.
3. SHAP interaction analysis.
4. Более глубокий анализ FN.
5. Feature engineering на основе FP/FN.
6. Сравнение моделей после threshold tuning.
7. Анализ fairness по группам клиентов.
8. Проверка stability модели.
9. Автоматическая генерация HTML/Markdown отчётов.
10. Model monitoring и data drift detection.

---

# 35. Итог

Финальная модель проекта:

```text
XGBoost
```

с threshold:

```text
0.236
```

Финальные результаты:

```text
Accuracy:  0.640
Precision: 0.446
Recall:    0.833
F1:        0.581
ROC-AUC:   0.792
```

Confusion matrix:

```text
TN = 78
FP = 62
FN = 10
TP = 50
```

Главные признаки по SHAP:

```text
Checking account
Duration
Credit amount
Monthly payment
Saving accounts
```

Проект показывает полный ML pipeline — от preprocessing и сравнения моделей до hyperparameter tuning, threshold optimization, model evaluation, feature importance, permutation importance и объяснения ошибок через SHAP.
