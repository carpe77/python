import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# 讀取數據
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 儲存 ID 欄位
train_id = train['Id']
test_id = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)

# 分離目標變數
y = train['SalePrice']
X = train.drop('SalePrice', axis=1)

# 合併訓練集和測試集進行差補
data = pd.concat([X, test], axis=0)

# 類別型特徵和連續型特徵分開
categorical_cols = data.select_dtypes(include=['object']).columns
numerical_cols = data.select_dtypes(exclude=['object']).columns

# 類別型數據處理：Label Encoding 並用數值替代
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = data[col].astype(str)  # 將 NaN 轉為字符串類型
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le  # 保存編碼器以便後續解碼

# KNN 差補
imputer = KNNImputer(n_neighbors=5)
data_imputed = imputer.fit_transform(data)

# 恢復為 DataFrame
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

# 恢復訓練集和測試集
X_train = data_imputed.iloc[:len(train), :]
X_test = data_imputed.iloc[len(train):, :]

# 切分訓練集和驗證集
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y, test_size=0.2, random_state=42)

# 使用 XGBoost 的原生接口進行訓練
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_test)

# 設置 XGBoost 參數
params = {
    'objective': 'reg:squarederror',  # 回歸問題
    'learning_rate': 0.05,
    'max_depth': 5,
    'eval_metric': 'rmse',           # 使用 RMSE 作為評估指標
    'random_state': 42
}

# 訓練模型並設置早停
evals = [(dtrain, 'train'), (dvalid, 'eval')]
model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=10,
    verbose_eval=False
)

# 預測並評估
y_pred = model.predict(dvalid)
rmse = mean_squared_error(y_valid, y_pred, squared=False)
print(f"驗證集 RMSE: {rmse:.2f}")

# 預測測試集
test_pred = model.predict(dtest)

# 儲存結果
submission = pd.DataFrame({'Id': test_id, 'SalePrice': test_pred})
submission.to_csv('submission.csv', index=False)
print("提交文件已保存為 submission.csv")
