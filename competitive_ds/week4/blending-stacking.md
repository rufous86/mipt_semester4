# **Blending и stacking**

## **Что такое блендинг и для чего он нужен**

Бывают ситуации, когда мы выжали из модели максимум, попробовали несколько моделей, и все равно нам не хватает несколько процентов, чтобы войти в лидеры. Блендинг превратить несколько моделей в одну самую сильную. При этом полученная модель будет иметь качество выше, чем каждая из моделей по отдельности. Если коротко, **блендинг** — это усреднение ответов нескольких моделей.

> **Основная идея блендинга** — взять от каждого алгоритма лучшее и совместить несколько разных ML-моделей в одну.

![image-20240322070057227](assets/image-20240322070057227.png)

**Что это дает:**

- Увеличение обобщающей способности финальной модели.
- Улучшение качества модели.
- Большую стабильность модели, что позволяет не слететь на приватном лидерборде.

Особенно хорошо накидывает блендинг, если смешиваемые **модели имеют разную природу**, например, нейронные сети, KNN и решающие деревья. Они выучивают разные зависимости и хорошо дополняют друг друга.

## Обучение моделей: CatBoost, LightGBM, XGBoost

Приступим к работе. Обучим поочередно три градиентных бустинга, а затем смешаем эти решения. Для этого сначала импортируем библиотеки:

```python
# Модели для смешивания
import lightgbm as lgbm
import xgboost as xgb
import catboost as cb
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")
```

Считываем данные:

```python
data = pd.read_csv("../data/quickstart_train.csv")

### Заменим категориальные признаки на числовые значения
cat_cols = ["model", "car_type", "fuel_type"]
for col in cat_cols:
    data[col] = data[col].replace(np.unique(data[col]), np.arange(data[col].nunique()))
    data[col] = data[col].astype("category")

data.head()
```

|       | **car_id** | **model** | **car_type** | **fuel_type** | **car_rating** | **year_to_start** | **riders** | **year_to_work** | **target_reg** | **target_class** | **mean_rating** | **distance_sum** | **rating_min** | **speed_max** | **user_ride_quality_median** | **deviation_normal_count** | **user_uniq** |
| ----- | ---------- | --------- | ------------ | ------------- | -------------- | ----------------- | ---------- | ---------------- | -------------- | ---------------- | --------------- | ---------------- | -------------- | ------------- | ---------------------------- | -------------------------- | ------------- |
| **0** | y13744087j | 8         | 1            | 1             | 3.78           | 2015              | 76163      | 2021             | 109.99         | another_bug      | 4.737759        | 1.214131e+07     | 0.1            | 180.855726    | 0.023174                     | 174                        | 170           |
| **1** | O41613818T | 23        | 1            | 1             | 3.90           | 2015              | 78218      | 2021             | 34.48          | electro_bug      | 4.480517        | 1.803909e+07     | 0.0            | 187.862734    | 12.306011                    | 174                        | 174           |
| **2** | d-2109686j | 16        | 3            | 1             | 6.30           | 2012              | 23340      | 2017             | 34.93          | gear_stick       | 4.768391        | 1.588366e+07     | 0.1            | 102.382857    | 2.513319                     | 174                        | 173           |
| **3** | u29695600e | 12        | 0            | 1             | 4.04           | 2011              | 1263       | 2020             | 32.22          | engine_fuel      | 3.880920        | 1.651883e+07     | 0.1            | 172.793237    | -5.029476                    | 174                        | 170           |
| **4** | N-8915870N | 16        | 3            | 1             | 4.70           | 2012              | 26428      | 2017             | 27.51          | engine_fuel      | 4.181149        | 1.398317e+07     | 0.1            | 203.462289    | -14.260456                   | 174                        | 171           |



Разделим выборку на валидационную и обучающую:

```python
cols2drop = ["car_id", "target_reg", "target_class"]

X_train, X_val, y_train, y_val = train_test_split(
    data.drop(cols2drop, axis=1),
    data["target_reg"],
    test_size=0.25,
    stratify=data["target_class"],
    random_state=42,
)
print(X_train.shape, X_val.shape)
(1752, 14) (585, 14)
```

### CatBoost

Теперь перейдем к непосредственному обучению моделей. Начнем с CatBoost:

```python
params_cat = {
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "depth": 3,
    "use_best_model": True,
    "cat_features": cat_cols,
    "text_features": [],
    # 'train_dir' : '/path/to/catboost/model',
    "border_count": 64,
    "l2_leaf_reg": 1,
    "bagging_temperature": 2,
    "rsm": 0.5,
    "loss_function": "RMSE",  # Не определена для регрессии
    # 'auto_class_weights' : 'Balanced', # Не определен для регрессии
    "random_state": 42,
    "custom_metric": ["MAE", "MAPE"],
}

cat_model = cb.CatBoostRegressor(**params_cat)
cat_model.fit(
    X_train,
    y_train,
    verbose=100,
    eval_set=(X_val, y_val),
    early_stopping_rounds=150,
)
0:	learn: 17.4391776	test: 17.9234161	best: 17.9234161 (0)	total: 46.2ms	remaining: 1m 9s
100:	learn: 12.0171853	test: 12.3281023	best: 12.3281023 (100)	total: 85.1ms	remaining: 1.18s
200:	learn: 11.4189213	test: 11.7777692	best: 11.7777692 (200)	total: 124ms	remaining: 803ms
300:	learn: 11.1134309	test: 11.6124163	best: 11.6124163 (300)	total: 161ms	remaining: 643ms
400:	learn: 10.8590320	test: 11.5378271	best: 11.5365214 (398)	total: 197ms	remaining: 541ms
500:	learn: 10.6685339	test: 11.5151383	best: 11.5129698 (494)	total: 232ms	remaining: 463ms
600:	learn: 10.5119473	test: 11.5108646	best: 11.4979901 (561)	total: 268ms	remaining: 401ms
700:	learn: 10.3486559	test: 11.5041313	best: 11.4913259 (636)	total: 305ms	remaining: 348ms
Stopped by overfitting detector  (150 iterations wait)

bestTest = 11.49132595
bestIteration = 636

Shrink model to first 637 iterations.

<catboost.core.CatBoostRegressor at 0x7ffa801e03d0>
```

После обучения получаем ответы на валиадационную выборку и оцениваем качество модели:

```python
print('MSE (catboost) : ', mean_squared_error(cat_model.predict(X_val), y_val).round(3))
MSE (catboost) :  132.051
# Сравним с бейзлайном в виде среднего значения
mean_squared_error(np.ones(len(y_val)) * y_val.mean(), y_val).round(3)
323.939
submit = pd.DataFrame({"target": cat_model.predict(X_val).reshape(-1)})
submit.to_csv("../tmp_data/catboost_preds.csv", index=False)
submit.head()
```

|       | **target** |
| ----- | ---------: |
| **0** |  32.987894 |
| **1** |  47.608527 |
| **2** |  35.089056 |
| **3** |  61.962611 |
| **4** |  71.456127 |



### LightGBM

Переходим к модели LightGBM. Сначала оптимизируем и обучим модель:

```python
params_lgbm = {
    "num_leaves": 200,
    "n_estimators": 1500,
    # "max_depth": 7,
    "min_child_samples": 2073,
    "learning_rate": 0.0051,
    "min_data_in_leaf": 10,
    "feature_fraction": 0.99,
    "categorical_feature": cat_cols,
    'reg_alpha' : 5.0,
    'reg_lambda' : 5.0,
}

lgbm_model = lgbm.LGBMRegressor(**params_lgbm)
lgbm_model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=100,
    verbose=150,
)
[LightGBM] [Warning] feature_fraction is set=0.99, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.99
[LightGBM] [Warning] min_data_in_leaf is set=10, min_child_samples=2073 will be ignored. Current value: min_data_in_leaf=10
Training until validation scores don't improve for 100 rounds
[150]	valid_0's l2: 189.37
[300]	valid_0's l2: 157.653
[450]	valid_0's l2: 144.558
[600]	valid_0's l2: 139.638
[750]	valid_0's l2: 139.828
Early stopping, best iteration is:
[674]	valid_0's l2: 139.132

LGBMRegressor(categorical_feature=['model', 'car_type', 'fuel_type'],
              feature_fraction=0.99, learning_rate=0.0051,
              min_child_samples=2073, min_data_in_leaf=10, n_estimators=1500,
              num_leaves=200, reg_alpha=5.0, reg_lambda=5.0)
print('MSE (lgb) : ', mean_squared_error(lgbm_model.predict(X_val), y_val).round(3))
MSE (lgb) :  139.132
submit = pd.DataFrame({"target": lgbm_model.predict(X_val).reshape(-1)})
submit.to_csv("../tmp_data/lgbm_preds.csv", index=False)
submit.head()
```

|       | **target** |
| ----- | ---------: |
| **0** |  33.660865 |
| **1** |  42.797476 |
| **2** |  34.142549 |
| **3** |  64.390769 |
| **4** |  66.186459 |



Видим, что MSE хуже, чем у CatBoost, — 139 (у CatBoost было 132, то есть ошибка меньше).

### XGBoost

Осталось обучить XGBoost:

```python
# XGBoost не умеет работать с категориальными признаками, так что нужно сделать ohe
X_train = pd.get_dummies(X_train, columns=["car_type", "fuel_type", "model"])
X_val = pd.get_dummies(X_val, columns=["car_type", "fuel_type", "model"])
X_train.shape
(1752, 43)
params_xgb = {
    "eta": 0.05,
    "max_depth": 5,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    'gamma': .01,
    'reg_lambda' : 0.1,
    'reg_alpha' : 0.5,
    "objective": "reg:linear",
    "eval_metric": "mae",
    'tree_method' : 'hist', # Supported tree methods for cat fs are `gpu_hist`, `approx`, and `hist`.
    'enable_categorical' : True
    
}

xgb_model = xgb.XGBRegressor(**params_xgb)
xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    early_stopping_rounds=100,
    verbose=25,
)
print('best_iteration', xgb_model.best_iteration)
[07:20:08] WARNING: ../src/objective/regression_obj.cu:213: reg:linear is now deprecated in favor of reg:squarederror.
[0]	validation_0-mae:42.24252	validation_1-mae:42.20440
[25]	validation_0-mae:12.70880	validation_1-mae:12.98292
[50]	validation_0-mae:7.79993	validation_1-mae:9.14011
[75]	validation_0-mae:6.97940	validation_1-mae:8.76001
[99]	validation_0-mae:6.55026	validation_1-mae:8.74984
best_iteration 90
print('MSE (xgboost) : ', mean_squared_error(xgb_model.predict(X_val), y_val).round(3))
MSE (xgboost) :  137.036
submit = pd.DataFrame({"target": xgb_model.predict(X_val).reshape(-1)})
submit.to_csv("../tmp_data/xgb_preds.csv", index=False)
submit.head()
```

|       | **target** |
| ----- | ---------- |
| **0** | 32.807278  |
| **1** | 43.922146  |
| **2** | 33.644127  |
| **3** | 60.386654  |
| **4** | 69.922684  |



MSE 137 — хуже, чем CatBoost, но лучше, чем LightGBM.

## Блендинг и принципы блендинга

Мы получили три модели. Теперь смешаем их. Быстрый и простой способ это сделать — **усреднить ответы**. Есть несколько способов это сделать. Мы сразу проведем **взвешивание** — смешивание с весами моделей.

```python
cb_model  = pd.read_csv("../tmp_data/catboost_preds.csv")["target"]  # 132.051 Лучшая (ставим ей больший вес)
xgb_model = pd.read_csv("../tmp_data/xgb_preds.csv")["target"]       # 137.844 Средняя 
lgb_model = pd.read_csv("../tmp_data/lgbm_preds.csv")["target"]      # 139.132 Худшая (ставим ей меньший вес)
```

Веса наших ответов в сумме должны быть равны единице. Так как CatBoost имеет лучшее качество, отдаем ему 50%, XGBoost — 35% и LightGBM — 15%:

```python
# score1 > score2 > score3 : w1 > w2 > w3
ensemble = cb_model * 0.50 + xgb_model * 0.35 + lgb_model * 0.15
```

Усредним ответы ансамбля и получим:

```python
print('MSE (ensemble) :', mean_squared_error(ensemble, y_val).round(3))
MSE (ensemble) : 131.85
round((100*(132.051 - 131.080)/132.051), 1)
0.7
```

Результат — скор улучшился. Ошибка ансамбля меньше, чем ошибка лучшей модели. При этом мы выжали меньше 1% качества, что неплохо.

**Как подбирать веса:**

- Опираться на скоры на лидерборде.
- Исходить из локальной валидации.
- Ставить веса пропорциональны скору.

Если хочется сделать кодом, то ниже более универсальный способ:

```python
weights = {"catboost": 0.5, "lgbm": 0.15, "xgb": 0.35}
import os

preds = pd.DataFrame()

# Соберем единый датафрейм из наших предсказаний
for model_name in ["catboost", "lgbm", "xgb"]:
    
    path = os.path.join("../tmp_data/", f"{model_name}_preds.csv")
    now = pd.read_csv(path).reset_index()

    now["model"] = model_name
    now["target"] *= weights[model_name]
    preds = pd.concat([preds, now])

preds.head()
```

|       | **index** | **target** | **model** |
| ----- | --------: | ---------: | --------: |
| **0** |         0 |  16.493947 |  catboost |
| **1** |         1 |  23.804264 |  catboost |
| **2** |         2 |  17.544528 |  catboost |
| **3** |         3 |  30.981306 |  catboost |
| **4** |         4 |  35.728063 |  catboost |



```python
preds["model"].unique()
array(['catboost', 'lgbm', 'xgb'], dtype=object)
ensemble = preds.groupby("index")["target"].agg("sum")
mean_squared_error(ensemble, y_val)
131.84991442662573
```

**Принципы блендинга:**

- Не бленди, пока не выжал максимум из моделей по отдельности.
- Чем различнее и сильнее модели, тем эффективнее блендинг (эффективность).
- При равной точности ансамбля побеждает соло-модель на привате (стабильность).
- Чем раньше проверим эффект от блендинга, тем эффективнее будет стратегия.
- Блендить можно с разными весами пропорционально скорам или разным фичам.
- Блендинг по фолдам и чекпоинтам обучения — это тоже блендинг.
- Блендинг по сидам — это стабилизирующий блендинг.
- Против блендинга только больший блендинг.

**Принципы стекинга:**

- Стекинг сильнее блендинга, но капризнее.
- Стекать можно с комбинации с исходными признаками: это позволяет сохранять контекст исходных данных.
- Стекинг бывает разных уровней, но это ресурсоемко. Дальше первого уровня уходят редко, дальше второго можно не заходить.
- Не можем больше стекать — блендим с решением сокомандников.

## Выводы

- Блендинг — это сильный инструмент, который зачастую неплохо поднимает качество моделей.
- При этом само смешивание провести можно вообще в одну строчку, просто взяв среднее моделей и взвесив.
- `ensemble = model1 * w1 + model2 * w2 + model3 * w3 + ...`

# **Automatic stacking**

## Три модели для блендинга

[Три модели для блендинга](https://github.com/a-milenkin/Competitive_Data_Science/blob/bd434f167731928739dce077d5ae9e9d2fa667ee/notebooks/#c1)

Чем больше моделей мы стекаем, тем код разрастается больше, а количество беспорядка растет по экспоненте. Но есть специальные инструменты, которые позволяют сделать это элегантно, более эффективно да еще и с меньшим числом строк кода. В этом модуле разберем Sklearn Pipelines.

**Sklearn Pipeline** — способ упаковать процесс обучения и инференса от Feature Engineering до стекинга десяти моделей в один пайплайн. К сожалению, и он имеет свои недостатки. Разберем подробнее.

Импортируем библиотеки:

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import lightgbm as lgbm
import xgboost as xgb
import catboost as cb
# pip install xgboost -U -q
import warnings
warnings.filterwarnings("ignore")
```

Считываем данные:

```python
from sklearn import preprocessing
data = pd.read_csv('../data/quickstart_train.csv')

categorical_features = ['model', 'car_type', 'fuel_type']

for cat in categorical_features:
    lbl = preprocessing.LabelEncoder()
    data[cat] = lbl.fit_transform(data[cat].astype(str))
    data[cat] = data[cat].astype('category')
    
data.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2337 entries, 0 to 2336
Data columns (total 17 columns):
 #   Column                    Non-Null Count  Dtype   
---  ------                    --------------  -----   
 0   car_id                    2337 non-null   object  
 1   model                     2337 non-null   category
 2   car_type                  2337 non-null   category
 3   fuel_type                 2337 non-null   category
 4   car_rating                2337 non-null   float64 
 5   year_to_start             2337 non-null   int64   
 6   riders                    2337 non-null   int64   
 7   year_to_work              2337 non-null   int64   
 8   target_reg                2337 non-null   float64 
 9   target_class              2337 non-null   object  
 10  mean_rating               2337 non-null   float64 
 11  distance_sum              2337 non-null   float64 
 12  rating_min                2337 non-null   float64 
 13  speed_max                 2337 non-null   float64 
 14  user_ride_quality_median  2337 non-null   float64 
 15  deviation_normal_count    2337 non-null   int64   
 16  user_uniq                 2337 non-null   int64   
dtypes: category(3), float64(7), int64(5), object(2)
memory usage: 264.2+ KB
```

Разделим выборку на валидационную и обучающую:

```python
# Значения таргета закодируем целыми числами
class_names = np.unique(data['target_class'])
data['target_class'] = data['target_class'].replace(class_names, np.arange(data['target_class'].nunique()))
cols2drop = ['car_id', 'target_reg', 'target_class']
categorical_features = ['model', 'car_type', 'fuel_type']
numerical_features = [c for c in data.columns if c not in categorical_features and c not in cols2drop]
X_train, X_val, y_train, y_val = train_test_split(data.drop(cols2drop, axis=1), 
                                                    data['target_class'],
                                                    test_size=.25,
                                                    stratify=data['target_class'],
                                                    random_state=42)
print(X_train.shape, X_val.shape)
(1752, 14) (585, 14)
```

Объявим три модели.

Модель CatBoost

```python
params_cat = {
             'n_estimators' : 700,
              # 'learning_rate': .03,
              'depth' : 3,
              'verbose': False,
              'use_best_model': True,
              'cat_features' : categorical_features,
              'text_features': [],
              # 'train_dir' : '/home/jovyan/work/catboost',
              'border_count' : 64,
              'l2_leaf_reg' : 1,
              'bagging_temperature' : 2,
              'rsm' : 0.51,
              'loss_function': 'MultiClass',
              'auto_class_weights' : 'Balanced', # Try not balanced
              'random_state': 42,
              'use_best_model': False,
              # 'custom_metric' : ['AUC', 'MAP'] # Не работает внутри sklearn.Pipelines
         }
cat_model = cb.CatBoostClassifier(**params_cat)
```

Модель LightGBM

```python
categorical_features_index = [i for i in range(data.shape[1]) if data.columns[i] in categorical_features]
params_lgbm = {
    "num_leaves": 200,
    "n_estimators": 1500,
    # "max_depth": 7,
    "min_child_samples": None,
    "learning_rate": 0.001,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.98,
    # "categorical_feature": cat_cols,
    'reg_alpha' : 3.0,
    'reg_lambda' : 5.0,
    'categorical_feature': categorical_features_index
}
lgbm_model = lgbm.LGBMClassifier(**params_lgbm)
```

Модель XGBoost

```python
params_xgb = {
    "eta": 0.05,
    'n_estimators' : 1500,
    "max_depth": 6,
    "subsample": 0.7,
    # "colsample_bytree": 0.95,
    'min_child_weight' : 0.1,
    'gamma': .01,
    'reg_lambda' : 0.1,
    'reg_alpha' : 0.5,
    "objective": "reg:linear",
    "eval_metric": "mae",
    'tree_method' : 'hist', # Supported tree methods for cat fs are `gpu_hist`, `approx`, and `hist`.
    'enable_categorical' : True
    
}
xgb_model = xgb.XGBClassifier(**params_xgb)
```

## Построение пайплайна

```python
!pip3 install -U scikit-learn==1.2.2
Collecting scikit-learn==1.2.2
  Using cached scikit_learn-1.2.2-cp310-cp310-macosx_12_0_arm64.whl (8.5 MB)
Requirement already satisfied: numpy>=1.17.3 in /Users/sergak/miniconda3/lib/python3.10/site-packages (from scikit-learn==1.2.2) (1.23.5)
Requirement already satisfied: scipy>=1.3.2 in /Users/sergak/miniconda3/lib/python3.10/site-packages (from scikit-learn==1.2.2) (1.10.1)
Requirement already satisfied: joblib>=1.1.1 in /Users/sergak/miniconda3/lib/python3.10/site-packages (from scikit-learn==1.2.2) (1.2.0)
Requirement already satisfied: threadpoolctl>=2.0.0 in /Users/sergak/miniconda3/lib/python3.10/site-packages (from scikit-learn==1.2.2) (3.1.0)
Installing collected packages: scikit-learn
  Attempting uninstall: scikit-learn
    Found existing installation: scikit-learn 1.3.0
    Uninstalling scikit-learn-1.3.0:
      Successfully uninstalled scikit-learn-1.3.0
Successfully installed scikit-learn-1.2.2

[notice] A new release of pip is available: 23.0.1 -> 23.1.2
[notice] To update, run: pip install --upgrade pip
# Вспомогательные блоки организации для пайплайна
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
# Вспомогательные элементы для наполнения пайплайна
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
# Некоторые модели для построения ансамбля
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
# Добавим визуализации
import sklearn
sklearn.set_config(display='diagram')

from warnings import simplefilter
# Ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
```

Предобработаем данные — под каждый тип данных заводим свой трансформер:

```python
# Заменяет пропуски самым частым значением и делает ohe
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
# Заменяет пропуски средним значением и делает нормализацию
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler())
])
# Соединим два предыдущих трансформера в один
preprocessor = ColumnTransformer(transformers=[
    ("numerical", numerical_transformer, numerical_features),
    ("categorical", categorical_transformer, categorical_features)])

preprocessor
ColumnTransformer(transformers=[('numerical',
                                 Pipeline(steps=[('imputer', SimpleImputer()),
                                                 ('scaler', StandardScaler())]),
                                 ['car_rating', 'year_to_start', 'riders',
                                  'year_to_work', 'mean_rating', 'distance_sum',
                                  'rating_min', 'speed_max',
                                  'user_ride_quality_median',
                                  'deviation_normal_count', 'user_uniq']),
                                ('categorical',
                                 Pipeline(steps=[('imputer',
                                                  SimpleImputer(strategy='most_frequent')),
                                                 ('onehot',
                                                  OneHotEncoder(handle_unknown='ignore'))]),
                                 ['model', 'car_type', 'fuel_type'])])
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. 
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
preprocessor.transformers[0]
('numerical',
 Pipeline(steps=[('imputer', SimpleImputer()), ('scaler', StandardScaler())]),
 ['car_rating',
  'year_to_start',
  'riders',
  'year_to_work',
  'mean_rating',
  'distance_sum',
  'rating_min',
  'speed_max',
  'user_ride_quality_median',
  'deviation_normal_count',
  'user_uniq'])
```

## Обучение ансамбля

Теперь обучим ансамбль:

```python
# Список базовых моделей
estimators = [
    
    
    ("ExtraTrees",  make_pipeline(preprocessor, ExtraTreesClassifier(n_estimators = 10_000, max_depth = 6, min_samples_leaf = 2, 
                                                              bootstrap = True, class_weight = 'balanced', # ccp_alpha = 0.001, 
                                                              random_state = 75, verbose=False, n_jobs=-1,))),
    

    ("XGBoost", xgb_model),
    ("LightGBM", lgbm_model),
    ("CatBoost", cat_model),
    
    # То, что не дало прироста в ансамбле
    # ("SVM", make_pipeline(preprocessor, LinearSVC(verbose=False))),
    # ("MLP", make_pipeline(preprocessor, MLPClassifier(verbose=False, hidden_layer_sizes=(100, 30, ), alpha=0.001,random_state=75, max_iter = 1300, ))),
    ("Random_forest",  make_pipeline(preprocessor, RandomForestClassifier(n_estimators = 15_000, max_depth = 7, 
                                                              min_samples_leaf = 2,
                                                              warm_start = True, n_jobs=-1,
                                                              random_state = 75, verbose=False))),
    
    
    
]

# В качестве мета-модели будем использовать LogisticRegression
meta_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(verbose=False),
    # final_estimator=RandomForestClassifier(n_estimators = 10_000, 
                                           # max_depth = 5,
                                           # verbose=False),
    n_jobs=-1,
    verbose=False,
)

stacking_classifier = meta_model
stacking_classifier
StackingClassifier(estimators=[('ExtraTrees',
                                Pipeline(steps=[('columntransformer',
                                                 ColumnTransformer(transformers=[('numerical',
                                                                                  Pipeline(steps=[('imputer',
                                                                                                   SimpleImputer()),
                                                                                                  ('scaler',
                                                                                                   StandardScaler())]),
                                                                                  ['car_rating',
                                                                                   'year_to_start',
                                                                                  			'riders',
                                                                                   'year_to_work',
                                                                                   'mean_rating',
                                                                                   'distance_sum',
                                                                                   'rating_min',
                                                                                   'speed_max',
                                                                                   'user_ride_quality_median',
                                                                                   'deviation_nor...
                                                                                                   SimpleImputer(strategy='most_frequent')),
                                                                                                  ('onehot',
                                                                                                   OneHotEncoder(handle_unknown='ignore'))]),
                                                                                  ['model',
                                                                                   'car_type',
                                                                                   'fuel_type'])])),
                                                ('randomforestclassifier',
                                                 RandomForestClassifier(max_depth=7,
                                                                        min_samples_leaf=2,
                                                                        n_estimators=15000,
                                                                        n_jobs=-1,
                                                                        random_state=75,
                                                                        verbose=False,
                                                                        warm_start=True))]))],
                   final_estimator=LogisticRegression(verbose=False), n_jobs=-1,
                   verbose=False)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. 
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
stacking_classifier.fit(X_train, y_train)
```

```markup
/Users/sergak/miniconda3/lib/python3.10/site-packages/lightgbm/basic.py:1487: UserWarning: categorical_feature keyword has been found in `params` and will be ignored.
Please use categorical_feature argument of the Dataset constructor to pass this parameter.
  _log_warning(f'{key} keyword has been found in `params` and will be ignored.\n'
/Users/sergak/miniconda3/lib/python3.10/site-packages/lightgbm/basic.py:1513: UserWarning: categorical_feature in param dict is overridden.
  _log_warning(f'{cat_alias} in param dict is overridden.')
/Users/sergak/miniconda3/lib/python3.10/site-packages/lightgbm/basic.py:1487: UserWarning: categorical_feature keyword has been found in `params` and will be ignored.
Please use categorical_feature argument of the Dataset constructor to pass this parameter.
  _log_warning(f'{key} keyword has been found in `params` and will be ignored.\n'
/Users/sergak/miniconda3/lib/python3.10/site-packages/lightgbm/basic.py:1513: UserWarning: categorical_feature in param dict is overridden.
  _log_warning(f'{cat_alias} in param dict is overridden.')
/Users/sergak/miniconda3/lib/python3.10/site-packages/lightgbm/basic.py:1487: UserWarning: categorical_feature keyword has been found in `params` and will be ignored.
Please use categorical_feature argument of the Dataset constructor to pass this parameter.
  _log_warning(f'{key} keyword has been found in `params` and will be ignored.\n'
/Users/sergak/miniconda3/lib/python3.10/site-packages/lightgbm/basic.py:1487: UserWarning: categorical_feature keyword has been found in `params` and will be ignored.
Please use categorical_feature argument of the Dataset constructor to pass this parameter.
  _log_warning(f'{key} keyword has been found in `params` and will be ignored.\n'
[LightGBM] [Warning] feature_fraction is set=0.98, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.98
[LightGBM] [Warning] feature_fraction is set=0.98, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.98
[LightGBM] [Warning] feature_fraction is set=0.98, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.98
[LightGBM] [Warning] feature_fraction is set=0.98, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.98
[LightGBM] [Warning] feature_fraction is set=0.98, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.98
[LightGBM] [Warning] feature_fraction is set=0.98, colsample_bytree=1.0 will be ignored. Current value: feature_fraction=0.98

StackingClassifier(estimators=[('ExtraTrees',
                                Pipeline(steps=[('columntransformer',
                                                 ColumnTransformer(transformers=[('numerical',
                                                                                  Pipeline(steps=[('imputer',
                                                                                                   SimpleImputer()),
                                                                                                  ('scaler',
                                                                                                   StandardScaler())]),
                                                                                  ['car_rating',
                                                                                   'year_to_start',
                                                                                   'riders',
                                                                                   'year_to_work',
                                                                                   'mean_rating',
                                                                                   'distance_sum',
                                                                                   'rating_min',
                                                                                   'speed_max',
                                                                                   'user_ride_quality_median',
                                                                                   'deviation_nor...
                                                                                                   SimpleImputer(strategy='most_frequent')),
                                                                                                  ('onehot',
                                                                                                   OneHotEncoder(handle_unknown='ignore'))]),
                                                                                  ['model',
                                                                                   'car_type',
                                                                                   'fuel_type'])])),
                                                ('randomforestclassifier',
                                                 RandomForestClassifier(max_depth=7,
                                                                        min_samples_leaf=2,
                                                                        n_estimators=15000,
                                                                        n_jobs=-1,
                                                                        random_state=75,
                                                                        verbose=False,
                                                                        warm_start=True))]))],
                   final_estimator=LogisticRegression(verbose=False), n_jobs=-1,
                   verbose=False)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. 
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
# For i in stacking_classifier.estimators:
#     Print(i[0])
# Dir(stacking_classifier)
corr_df = pd.DataFrame()

for model, (name, _) in zip(stacking_classifier.estimators_, stacking_classifier.estimators):
    preprocessed = stacking_classifier.estimators[0][1].steps[0][1].fit(X_train, y_train).transform(X_val)
    print(name, 'accuracy: ', round(accuracy_score(model.predict(X_val), y_val), 4))
    
    corr_df[name] = model.predict(X_val)
ExtraTrees accuracy:  0.7368
XGBoost accuracy:  0.7949
LightGBM accuracy:  0.8085
CatBoost accuracy:  0.8034
Random_forest accuracy:  0.7487
corr_df.corr().style.background_gradient(cmap="RdYlGn")
```

|                   | **ExtraTrees** | **XGBoost** | **LightGBM** | **CatBoost** | **Random_forest** |
| ----------------- | -------------: | ----------: | -----------: | -----------: | ----------------: |
| **ExtraTrees**    |       1.000000 |    0.856435 |     0.851740 |     0.855011 |          0.841299 |
| **XGBoost**       |       0.856435 |    1.000000 |     0.976487 |     0.953732 |          0.946774 |
| **LightGBM**      |       0.851740 |    0.976487 |     1.000000 |     0.969285 |          0.953330 |
| **CatBoost**      |       0.855011 |    0.953732 |     0.969285 |     1.000000 |          0.948848 |
| **Random_forest** |       0.841299 |    0.946774 |     0.953330 |     0.948848 |          1.000000 |



```python
# Random_forest сильно коррелирует с другими моделями, 
# Поэтому он снижает качество ансамбля
# Попробуйте его убрать
print('ensemble score:', round(accuracy_score(stacking_classifier.predict(X_val), y_val), 4))
# Как вы можете заметить, ансамбль довольно заметно улучшил качество решения
ensemble score: 0.8051
```

**Комментарии:**

- Да, скор ансамбля вырос, но у этой реализации есть много «но».
- В качестве метамодели использовалась `LogisticRegression`, что, по сути, обычный блендинг с кросс-валидацией.
- Слабые или похожие модели мешают ансамблю поднять скор (если убрать `RandomForest`, то скор поднимется).
- Стекинг можно усложнить, подавая мета-модели еще признаки, при этом используя более сложную мета-модель. Тогда в зависимости от свойств объекта мета-модели, такие как `RandomForestClassifier`, могут принимать решение оптимальнее.
- В рамках `pipeline` в `sklearn` это сделать сложнее. Надо взять что-то другое.
- Не все можно поместить в `pipeline`. Например, `eval_set` для `early-stopping` или класс `train` от LightGBM.

## Выводы

- Sklearn Pipeline — это очень сильный инструмент, позволяющий упаковать весь процесс обучения модели в один механизм.
- Инструмент помогает добавлять новые модели, которые легко применять на инференсе.
- Sklearn Pipeline часто используется не только на соревнованиях, но и в обычной работе из-за своей элегантности и простоты.



## Литература для дополнительного изучения

- [А. Дьяконов. Анализ малых данных](https://alexanderdyakonov.wordpress.com/2017/03/10/cтекинг-stacking-и-блендинг-blending/)
- [Носков, Котик, Галицкий. Kaggle Toxic Comment: выявление и классификация токсичных комментариев: пример решения](https://www.youtube.com/watch?v=aMlpeDOjib8)
