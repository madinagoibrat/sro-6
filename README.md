from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Создаем имитационный набор данных
data = {
    'Возраст': [25, 30, 35, 20, 28, 32, 45, 18],
    'Посещения': [10, 5, 8, 2, 7, 12, 15, 1],
    'Покупка': [1, 0, 1, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)

# Разделяем признаки и целевую переменную
X = df[['Возраст', 'Посещения']]
y = df['Покупка']

# Создаем и обучаем модель дерева решений
model = DecisionTreeClassifier()
model.fit(X, y)

# Теперь мы можем использовать модель для предсказания покупок
# Предположим, у нас есть новый клиент, 22 года, с 6 посещениями сайта
new_customer = [[22, 6]]
prediction = model.predict(new_customer)

if prediction[0] == 1:
    print("Этот клиент совершит покупку.")
else:
    print("Этот клиент не совершит покупку.")
