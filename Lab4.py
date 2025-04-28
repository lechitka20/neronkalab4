import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd


# Cоздадим простую нейронную сеть.
# Для этого объявим свой класс 
# наш класс будем наследовать от nn.Module, который включает большую часть необходимого нам функционала

class NNet(nn.Module):
    # для инициализации сети на вход нужно подать размеры (количество нейронов) входного, скрытого и выходного слоев
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        # nn.Sequential - контейнер модулей
        # он последовательно объединяет слои и позволяет запускать их одновременно
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size), # слой линейных сумматоров
                                    nn.Tanh(),                       # функция активации
                                    nn.Linear(hidden_size, out_size),
                                    nn.Tanh()
                                    )
    # прямой проход    
    def forward(self,X):
        pred = self.layers(X)
        return pred



# Данные как и в прошлых работах загружаем и выделяем в отдельные переменные
# X - признаки
# y - правильные ответы, их кодируем числами
# X и y преобразуем в pytorch тензоры
df = pd.read_csv('data.csv')
X = torch.Tensor(df.iloc[0:100, 0:3].values)
y = df.iloc[0:100, 4].values
y = torch.Tensor(np.where(y == "Iris-setosa", 1, -1).reshape(-1,1))


# Параметры нашей сети определяются данными.
# Размер входного слоя - это количество признаков в задаче, т.е. количество 
# столбцов в X.
inputSize = X.shape[1] # количество признаков задачи 

# Размер (количество нейронов) в скрытом слое задается нами, четких правил как выбрать
# этот параметр нет, это открытая проблема в нейронных сетях.
# Но есть общий принцип - чем сложнее зависимость (разделяющая поверхность), 
# тем больше нейронов должно быть в скрытом слое.
hiddenSizes = 3 #  число нейронов скрытого слоя 

# Количество выходных нейронов равно количеству классов задачи.
# Но для двухклассовой классификации можно задать как один, так и два выходных нейрона.
outputSize = 1


# Создаем экземпляр нашей сети
net = NNet(inputSize,hiddenSizes,outputSize)

# Веса нашей сети содержатся в net.parameters() 
for param in net.parameters():
    print(param)

# Можно вывести их с названиями
for name, param in net.named_parameters():
    print(name, param)


# Посчитаем ошибку нашего не обученного алгоритма
# градиенты нужны только для обучения, тут их можно отключить, 
# это немного ускорит вычисления
with torch.no_grad():
    pred = net.forward(X)

# Так как наша сеть предсказывает числа от -1 до 1, то ее ответы нужно привести 
# к значениям меток
pred = torch.Tensor(np.where(pred >=0, 1, -1).reshape(-1,1))

# Считаем количество ошибочно классифицированных примеров
err = sum(abs(y-pred))/2
print(err) # до обучения сеть работает случайно, как бросание монетки

# Для обучения нам понадобится выбрать функцию вычисления ошибки
lossFn = nn.MSELoss()

# и алгоритм оптимизации весов
# при создании оптимизатора в него передаем настраиваемые параметры сети (веса)
# и скорость обучения
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# В цикле обучения "прогоняем" обучающую выборку
# X - признаки
# y - правильные ответы
# epohs - количество итераций обучения

epohs = 100
for i in range(0,epohs):
    pred = net.forward(X)   #  прямой проход - делаем предсказания
    loss = lossFn(pred, y)  #  считаем ошибу 
    optimizer.zero_grad()   #  обнуляем градиенты 
    loss.backward()
    optimizer.step()
    if i%10==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

    
# Посчитаем ошибку после обучения
with torch.no_grad():
    pred = net.forward(X)

pred = torch.Tensor(np.where(pred >=0, 1, -1).reshape(-1,1))
err = sum(abs(y-pred))/2
print('\nОшибка (количество несовпавших ответов): ')
print(err) # обучение работает, не делает ошибок или делает их достаточно мало

###############################################################################
###############################################################################
# Но что делать, если нужно предсказывать не два, а больше классов?
# При количестве классов больше двух нам нужно по другому кодировать метки классов.
# Теперь на каждый класс у нас должен быть свой нейрон, соответственно 
# переменная, содержащая ответы теперь будет не вектором, а матрицей.

df = pd.read_csv('data_3class.csv')
X = torch.Tensor(df.iloc[:, 0:3].values) # Признаки остаются без изменений
y = df.iloc[:, 4].values                # ответы берем из четвертого столбца как и раньше
# Но теперь классы кодируются иначе.
# Метка класса будет состоять из 3-х бит, где каждому классу соответствует одна единица
# такой подход называтеся one-hot encoding
print(pd.get_dummies(y).iloc[0:150:50,:])

labels = pd.get_dummies(y).columns.values  #  отдельно сохраним названия классов
y = torch.Tensor(pd.get_dummies(y).values) # сохраним значения


# в структуру нашей сети необходимо внести изменения
# гиперболический тангенс в выходном слое теперь нам не подходит,
# т.к. мы ожидаем 0 или 1 на выходе нейронов, то нам подойдет Сигмоида в качестве
# функции активации выходного слоя
class NNet_multiclass(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size), # слой линейных сумматоров
                                    nn.Tanh(),                       # функция активации
                                    nn.Linear(hidden_size, out_size),
                                    nn.Sigmoid(),
                                    # nn.Softmax(dim=1) # вместо сигмоиды можно использовать softmax
                                    )
    # прямой проход    
    def forward(self,X):
        pred = self.layers(X)
        return pred

# задаем параметры сети
inputSize = X.shape[1] # количество признаков задачи 
hiddenSizes = 3     #  число нейронов скрытого слоя 
outputSize = y.shape[1] # число нейронов выходного слоя равно числу классов задачи

net = NNet_multiclass(inputSize,hiddenSizes,outputSize)

# В задачах многоклассовой классификации используется ошибка,
# вычисляющая разницу между предсказанной вероятностью появления класса 
# и истинной вероятностью его появления для конкретного примера
lossFn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

epohs = 100
for i in range(0,epohs):
    pred = net.forward(X)   #  прямой проход - делаем предсказания
    loss = lossFn(pred, y)  #  считаем ошибу 
    optimizer.zero_grad()   #  обнуляем градиенты 
    loss.backward()
    optimizer.step()
    if i%10==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

    
# Посчитаем ошибку после обучения
with torch.no_grad():
    pred = net.forward(X)

pred_lbl = labels[pred.max(1).indices] # предсказанные названия классов
true_lbl = labels[y.max(1).indices]    # истинные названия классов

err = sum(pred_lbl!=true_lbl) # все что не совпало, считаем ошибками
print('\nОшибка (количество несовпавших ответов): ')
print(err)                    # ошибок много, но попробуйте увеличить число скрытых нейронов


###############################################################################
###############################################################################
# Теперь посмотрим что изменится в структуре нейронной сети, если нам нужно решить задачу регрессии
# Задача регрессии заключается в предсказании значений одной переменной
# по значениям другой (других).  
# От задачи классификации отличается тем, что выходные значения нейронной сети не 
# ограничиваются значениями меток классов (0 или 1), а могут лежать в любом 
# диапазоне чисел.
# Примерами такой задачи можгут быть предсказание цен на жилье, курсов валют или акций,
# количества выпадающих осадков или потребления электроэнергии.

# Рассмотрим задачу предсказания прочности бетона (измеряется в мегапаскалях)
df = pd.read_csv('concrete_data.csv')

# Известно что прочность бетона зависит от многих факторов - количесва цемента, 
# используемых добавок, 
# Cement - количество цемента в растворе kg/m3
# Blast Furnace Slag - количество шлака в растворе kg/m3 
# Fly Ash - количетво золы в растворе kg/m3
# Water - количетво воды в растворе kg/m3
# Superplasticizer - количетво пластификатора в растворе kg/m3
# Coarse Aggregate - количетво крупного наполнителя в растворе kg/m3
# Fine Aggregate - количетво мелкого наполнителя в растворе kg/m3
# Age - возраст бетона в днях
# Concrete compressive strength -  прочность бетона MPa


X = torch.Tensor(df.iloc[:, [0]].values) # выделяем признаки (независимые переменные)
y = torch.Tensor(df.iloc[:, -1].values)  #  предсказываемая переменная, ее берем из последнего столбца

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(df.iloc[:, [0]].values, df.iloc[:, -1].values, marker='o')

# Чтобы выходные значения сети лежали в произвольном диапазоне,
# выходной нейрон не должен иметь функции активации или 
# фуннкция активации должна иметь область значений от -бесконечность до +бесконечность

class NNet_regression(nn.Module):
    
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(in_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, out_size) # просто сумматор
                                    )
    # прямой проход    
    def forward(self,X):
        pred = self.layers(X)
        return pred

# задаем параметры сети
inputSize = X.shape[1] # количество признаков задачи 
hiddenSizes = 3   #  число нейронов скрытого слоя 
outputSize = 1 # число нейронов выходного слоя

net = NNet_regression(inputSize,hiddenSizes,outputSize)

# В задачах регрессии чаще используется способ вычисления ошибки как разница квадратов
# как усредненная разница квадратов правильного и предсказанного значений (MSE)
# или усредненный модуль разницы значений (MAE)
lossFn = nn.L1Loss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.0001)

epohs = 1
for i in range(0,epohs):
    pred = net.forward(X)   #  прямой проход - делаем предсказания
    loss = lossFn(pred.squeeze(), y)  #  считаем ошибу 
    optimizer.zero_grad()   #  обнуляем градиенты 
    loss.backward()
    optimizer.step()
    if i%1==0:
       print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

    
# Посчитаем ошибку после обучения
with torch.no_grad():
    pred = net.forward(X)

print('\nПредсказания:') # Иногда переобучается, нужно запускать обучение несколько раз
print(pred[0:10])
err = torch.mean(abs(y - pred.T).squeeze()) # MAE - среднее отклонение от правильного ответа
print('\nОшибка (MAE): ')
print(err) # измеряется в MPa


# Построим график
plt.figure()
plt.scatter(df.iloc[:, [0]].values, df.iloc[:, -1].values, marker='o')

with torch.no_grad():
    y1 = net.forward(torch.Tensor([100]))
    y2 = net.forward(torch.Tensor([600]))

plt.plot([100,600], [y1.numpy(),y2.numpy()],'r')


#Задание. Логинов, 23ВП2, вариант 17
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных из CSV-файла
df = pd.read_csv('dataset_simple.csv')

# Подготовка данных:
# X - матрица признаков (возраст и доход)
# y - вектор целевых значений (метки классов)
X = df.iloc[:, :2].values  # Выбираем первые два столбца как признаки
y = df.iloc[:, 2].values   # Третий столбец содержит метки классов

# Преобразование данных в тензоры PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)  # Тензор признаков
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Тензор меток классов

# Нормализация данных для улучшения сходимости обучения
mean = X_tensor.mean(dim=0)  # Вычисляем средние значения признаков
std = X_tensor.std(dim=0)    # Вычисляем стандартные отклонения

# Защита от деления на ноль
std[std == 0] = 1e-8  # Заменяем нулевые отклонения малым значением

# Применяем нормализацию
X_tensor_normalized = (X_tensor - mean) / std

# Определение архитектуры нейронной сети
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        # Последовательная модель:
        # 1. Полносвязный слой с 2 входами и 8 нейронами
        # 2. Функция активации ReLU
        # 3. Выходной слой с 1 нейроном
        # 4. Сигмоида для бинарной классификации
        self.layers = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Прямой проход через все слои
        return self.layers(x)

# Инициализация модели
model = SimpleClassifier()

# Настройка функции потерь и оптимизатора
criterion = nn.BCELoss()  # Бинарная кросс-энтропия
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Оптимизатор Adam

# Параметры обучения
num_epochs = 500  # Общее количество эпох
loss_history = []  # Для записи истории ошибок

# Процесс обучения
for epoch in range(num_epochs):
    # Прямой проход
    outputs = model(X_tensor_normalized)
    loss = criterion(outputs, y_tensor)
    
    # Обратное распространение и оптимизация
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Сохранение текущей ошибки
    loss_history.append(loss.item())
    
    # Логирование процесса обучения
    if (epoch + 1) % 50 == 0:
        print(f'Эпоха [{epoch + 1}/{num_epochs}], Ошибка: {loss.item():.4f}')

# Получение предсказаний после обучения
with torch.no_grad():
    predicted = model(X_tensor_normalized).numpy()
    predicted_classes = np.where(predicted >= 0.5, 1, 0)  # Преобразование вероятностей в классы

# Визуализация границы решений
plt.figure(figsize=(8, 6))
# Определение границ графика
x_min, x_max = X_tensor_normalized[:, 0].min() - 0.5, X_tensor_normalized[:, 0].max() + 0.5
y_min, y_max = X_tensor_normalized[:, 1].min() - 0.5, X_tensor_normalized[:, 1].max() + 0.5

# Создание сетки для визуализации
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# Преобразование сетки в формат тензора
grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

# Получение предсказаний для всей сетки
with torch.no_grad():
    grid_predictions = model(grid_tensor).numpy().reshape(xx.shape)

# Визуализация областей решений
plt.contourf(xx, yy, grid_predictions, alpha=0.4, cmap=plt.cm.RdBu, levels=np.linspace(0, 1, 21))
plt.contour(xx, yy, grid_predictions, levels=[0.5], colors='black', linewidths=1.5)

# Отображение реальных данных
plt.scatter(X_tensor_normalized[y == 0, 0], X_tensor_normalized[y == 0, 1], 
            color='red', label='Не купит', edgecolor='k')
plt.scatter(X_tensor_normalized[y == 1, 0], X_tensor_normalized[y == 1, 1], 
            color='blue', label='Купит', edgecolor='k')

# Настройка графика
plt.title('Граница решений классификатора')
plt.xlabel('Нормализованный возраст')
plt.ylabel('Нормализованный доход')
plt.legend()
plt.show()

# Расчет точности модели
accuracy = (predicted_classes.flatten() == y).mean()
print(f'Точность модели: {accuracy:.2%}')

# Визуализация процесса обучения
plt.figure(figsize=(8, 4))
plt.plot(loss_history, label='Значение функции потерь', color='orange')
plt.title('Динамика изменения ошибки')
plt.xlabel('Номер эпохи')
plt.ylabel('Ошибка')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
































































































# Пасхалка, кто найдет и сможет объяснить, тому +
# X = np.hstack([np.ones((X.shape[0], 1)), df.iloc[:, [0]].values])

# y = df.iloc[:, -1].values

# w = np.linalg.inv(X.T @ X) @ X.T @ y

# predicted = X @ w

# print(predicted)


























