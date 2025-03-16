import numpy as np
import time
# pip install matplotlib
import matplotlib.pyplot as plt
import random

# Функция для вычисления среднего арифметического с использованием NumPy
def calculate_average_numpy(matrix):
    sub_matrix = matrix[1:, 1:3]
    average = np.mean(sub_matrix)
    return average

# Функция для вычисления среднего арифметического со списком
def calculate_average_list(matrix):
    # Извлекаем подматрицу
    submatrix = [row[1:3] for row in matrix[1:]]
    # Создаем список всех элементов подматрицы
    elements = [element for row in submatrix for element in row]
    # Вычисляем среднее арифметическое
    average = sum(elements) / len(elements)
    return average

def measure_time(func, matrix, repeats=5):
    """
    Замеряет время выполнения функции func
    Выполняет функцию заданное кол-во раз и возвращает среднее, минимальное и максимальное время выполнения.

    func - это функция, время выполнения которой мы хотим измерить
    """
    if repeats <= 0:
        raise ValueError("Количество повторений должно быть больше 0")

    times = []
    result = None  # Инициализируем переменную result

    for _ in range(repeats):
        start_time = time.time()
        result = func(matrix)  # Выполняем функцию
        end_time = time.time()
        times.append(end_time - start_time)
    return result, np.mean(times), min(times), max(times)

# Размеры матриц
sizes = [5, 10, 50, 100, 1000]

# Список результирующего времени
res_list = []

# Результаты замеров
times_numpy_mean = []
times_numpy_min = []
times_numpy_max = []

times_list_mean = []
times_list_min = []
times_list_max = []

for size in sizes:
    # Генерация матрицы
    N, M = size, size
    A = np.random.randint(1, 10, size=(N, M))
    # Преобразуем NumPy массив в список
    A_list = A.tolist()

    # Замер времени для NumPy
    average_numpy, mean_time_numpy, min_time_numpy, max_time_numpy = measure_time(calculate_average_numpy, A, repeats=5)
    times_numpy_mean.append(mean_time_numpy)
    times_numpy_min.append(min_time_numpy)
    times_numpy_max.append(max_time_numpy)
    # Среднее значение по строкам (Задание 1)
    start_time = time.time()
    row_means = np.mean(A, axis=1)
    # Макс из сред. (Задание 1)
    max_mean = np.max(row_means)
    end_time = time.time()
    res_time = end_time - start_time
    res_list.append(res_time)
    # Среднее значение по строкам (Задание 1)
    start_time = time.time()
    row_means1 = [sum(row) / len(row) for row in A_list]
    # Макс из сред. (Задание 1)
    max_mean1 = max(row_means1)
    end_time = time.time()
    res_time1 = end_time - start_time
    res_list.append(res_time1)
    # Вычисление минимального (Задание 2)
    start_time = time.time()
    argmax = np.argmax(A)
    minimal = np.min(argmax)
    end_time = time.time()
    res_time2 = end_time - start_time
    res_list.append(res_time2)
    # Вычисление среднего по столбцам (Задание 3)
    start_time = time.time()
    col_mean = np.mean(A, axis=0)
    min_mean = np.min(col_mean)
    end_time = time.time()
    res_time3 = end_time - start_time
    res_list.append(res_time3)
    # (Задание 4)
    start_time = time.time()
    col_means = np.mean(A, axis=0)
    row_means = np.mean(A, axis=1)
    matrix_means = np.mean(A)
    end_time = time.time()
    res_time4 = end_time - start_time
    res_list.append(res_time4)
    # (Задание 5)
    start_time = time.time()
    h = 76
    V, B = 0, 0
    V = size // 5
    B = size // 5
    pod_mat = A[0:V, 0:B]
    result = pod_mat * h
    end_time = time.time()
    res_time5 = end_time - start_time
    res_list.append(res_time5)
    
    # Замер времени для списка
    average_list, mean_time_list, min_time_list, max_time_list = measure_time(calculate_average_list, A_list, repeats=5)
    times_list_mean.append(mean_time_list)
    times_list_min.append(min_time_list)
    times_list_max.append(max_time_list)

    # Вывод результатов
    print(f"Размер матрицы: {size}x{size}")
    print(f"Среднее арифметическое для NumPy массива: {average_numpy}")
    print(f"Среднее время выполнения для NumPy массива: {mean_time_numpy:.7f} секунд")
    print(f"Минимальное время выполнения для NumPy массива: {min_time_numpy:.7f} секунд")
    print(f"Максимальное время выполнения для NumPy массива: {max_time_numpy:.7f} секунд")
    print(f"(Задание№1) Наибольшее среди средних значений для NumPy массива:{max_mean}")
    print(f"(Задание№1) Время выполнения: {res_time} секунд")
    print(f"(Задание№2) Наименьший элемент строки матрицы А, для которого сумма абсолютных значений элементов максимальна:{minimal}")
    print(f"(Задание№2) Время выполнения: {res_time2} секунд")
    print(f"(Задание№3) Наименьший элемент по столбцам: {min_mean}")
    print(f"(Задание№3) Время выполнения: {res_time3}")
    print(f"(Задание№4) Среднее значенине по столбцам: {col_means}")
    print(f"(Задание№4) Среднее значенине по рядам: {row_means}")
    print(f"(Задание№4) Среднее значенине по матрице: {matrix_means}")
    print(f"(Задание№4) Время выполнения:{res_time4} секунд")
    print(f"(Задание№5) Подматрица(flattened) помноженная на 76:\n{result.flatten()}")
    print(f"(Задание№5) Время выполнения:{res_time5} секунд")

    
    
    print(f"Среднее арифметическое для списка: {average_list}")
    print(f"Среднее время выполнения для списка: {mean_time_list:.7f} секунд")
    print(f"Минимальное время выполнения для списка: {min_time_list:.7f} секунд")
    print(f"Максимальное время выполнения для списка: {max_time_list:.7f} секунд")
    print(f"(Задание№1) Наибольшее среди средних значений для списка:{max_mean1}")
    print(f"(Задание№1) Время выполнения: {res_time1} секунд")

# Построение графиков

# Создаем фигуру (окно для графика) с размером 10x6 дюймов
plt.figure(figsize=(10, 6))

# Строим график для данных times_numpy
# sizes — это список размеров матриц (ось X)
# times_numpy — это список времени выполнения для NumPy
# label="NumPy" — подпись для графика (появится в легенде)
# marker="o" — маркер в виде кружков для точек на графике

# Строим график для среднего времени выполнения NumPy
plt.plot(sizes, times_numpy_mean, label="Numpy (среднее)", marker="o")

plt.xlabel("Размер матрицы (N x N)")
plt.ylabel("Время выполнения (секунды)")

plt.title("Зависимость времени выполнения от размера матрицы")

plt.legend()
plt.grid(True)

plt.xscale("linear")
plt.yscale("linear")


# Строим график для данных times_list
# sizes — это список размеров матриц (ось X)
# times_list — это список времени выполнения для списка (ось Y)
# label="Список" — подпись для графика (появится в легенде)
# marker="o" — маркер в виде кружков для точек на графике

# Строим график для среднего времени выполнения списка
plt.plot(sizes, times_list_mean, label="Список (среднее)", marker="o")

# Подписываем оси X и Y
plt.xlabel("Размер матрицы (N x N)")
plt.ylabel("Время выполнения (секунды)")

# Заголовок графика
plt.title("Зависимость времени выполнения от размера матрицы")

# Добавляем легенду (подписи к графикам). Легенда показывает подписи к каждому графику (в данном случае "NumPy" и "Список")
plt.legend()

# Включаем сетку на графике. Сетка помогает визуально оценить значения на графике
plt.grid(True)

'''
    Логарифмическая шкала используется для визуализации данных, которые охватывают несколько порядков величины (например, от 1 до 10,000). 
    Она помогает сделать график более читаемым и информативным, когда значения на оси X или Y изменяются экспоненциально или очень сильно 
    различаются

    Если данные на графике изменяются в широком диапазоне (например, от 0.001 до 1000), то на линейной шкале:
    Малые значения будут "сливаться" и станут неразличимыми.
    Большие значения могут "перекосить" график, сделав его менее информативным.

    Логарифмическая шкала "сжимает" большие значения и "растягивает" малые, что позволяет одновременно видеть и малые, и большие значения
'''

# Устанавливаем логарифмическую шкалу для осей X и Y
plt.xscale("linear")
plt.yscale("linear") #хоть линейные графики и сливаются, но тут можно увидеть общую прогрессию, в случае с логарифмом получается что-то совсем не информативное
#plt.xscale("log")
#plt.yscale("log")
# Отображаем график
plt.show()

