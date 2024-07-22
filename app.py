import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 創建一個簡單的數據集
data = {
    'Feature1': [2.3, 1.7, 3.1, 3.5, 2.1, 1.6, 2.8, 3.0, 3.2, 2.7],
    'Feature2': [4.5, 3.2, 5.1, 5.5, 3.9, 2.4, 4.3, 4.8, 5.0, 4.1],
    'Label': [0, 0, 1, 1, 0, 0, 1, 1, 1, 0]
}

# 轉換為DataFrame
df = pd.DataFrame(data)

# 分離特徵和標籤
X = df[['Feature1', 'Feature2']]
y = df['Label']

# 切分訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化邏輯回歸模型
model = LogisticRegression()

# 訓練模型
model.fit(X_train, y_train)

# 使用測試集進行預測
y_pred = model.predict(X_test)

# 評估模型
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# 輸出結果
print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
