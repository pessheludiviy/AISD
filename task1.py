'''
    1.	Выполнить обработку элементов матрицы А, имеющей N строк и М столбцов. 
        Найти наибольшее значение среди средних значений для каждой строки матрицы. Сделать с помощью numpy (mean и max) и без него. 
        Замерить время выполнения программы.
        
    2. Выполнить обработку элементов матрицы А, имеющей N строк и М столбцов. Найти наименьший элемент строки матрицы А, 
        для которого сумма абсолютных значений элементов максимальна (np.argmax и np.min). Замерить время выполнения программы.
        
    3. Выполнить обработку элементов матрицы А, имеющей N строк и М столбцов. 
        Найти наименьшее значение среди средних значений для каждого столбца матрицы. Замерить время выполнения программы.
        
    4. Выполнить обработку элементов матрицы А, имеющей N строк и М столбцов. 
        Определить средние значения по всем строкам, столбцам матрицы и по всей матрицы. Замерить время выполнения программы.
        
    5. Выполнить обработку элементов матрицы A, имеющей N строк и M столбцов. 
        Каждый элемент подматрицы, умножить на заданное число. Замерить время выполнения программы.

'''

import numpy as np
import time
import random


print(f"========Своё решение==========\n")


N = 10000 
M = 10000

A = np.random.randint(0, 100, size=(N, M))

 
start_time = time.time()

row_means = np.mean(A, axis=1)

max_mean = np.max(row_means)

end_time = time.time()


print("Матрица A:")
print(A)
print("Список средних значений строк:", row_means)
print("Наибольшее значение из средних:", max_mean)
print(f"рантайм с нампай: {(end_time - start_time):.4f} сек")

a = 0
while a<3:
    print('\n')
    a+=1
    
#РЕШЕНИЕ БЕЗ NUMPY
print()
N1 = 10000 
M1 = 10000 


A1 = [[random.randint(0, 100) for _ in range(M1)] for _ in range(N1)]

start_time1 = time.time()


row_means1 = [sum(row) / len(row) for row in A]


max_mean1 = max(row_means1)

end_time1 = time.time()

#print("Матрица A:")
#for row in A1:
#    print(row)
print("Средние значения каждой строчки:", row_means1[:10])
print("Максимальное среди средних:", max_mean1)
print(f"Рантайм без нампай: {(end_time1 - start_time1):.4f} сек")

a = 0
while a<3:
    print('\n')
    a+=1

print("==========Задание№2============")

K = 10000
L = 10000
B = np.random.randint(0, 100, size = (K, L))
start_time2 = time.time()
argmax = np.argmax(B)
minimal = np.min(argmax)
end_time2 = time.time()
print(f"наименьший элемент строки матрицы А, для которого сумма абсолютных значений элементов максимальна:\n{minimal}")
print(f"Рантайм:{(end_time2 - start_time2):.4f} сек.")

a = 0
while a<3:
    print('\n')
    a+=1

print("==========Задание№3============")
Q, W = 10000, 10000
C = np.random.randint(0, 100, size=(Q, W))
start_time3 = time.time()
col_means = np.mean(C, axis=0)
min_mean = min(col_means)
end_time3 = time.time()
print(f"Минимальное значение среди средних значений столбцов матрицы:{min_mean}")
print(f"Рантайм:{(end_time3 - start_time3):.4f}")

a = 0
while a<3:
    print('\n')
    a+=1

print("==========Задание№4============")
E, R = 10000, 10000
D = np.random.randint(0, 100, size=(E, R))
start_time4 = time.time()
col_means = np.mean(D, axis=0)
row_means = np.mean(D, axis=1)
matrix_means = np.mean(D)
end_time4 = time.time()
print(f"Минимальное значенине по столбцам: {col_means}")
print(f"Минимальное значенине по рядам: {row_means}")
print(f"Минимальное значенине по матрице: {matrix_means}")
print(f"Рантайм:{(end_time4 - start_time4):.4f} сек")

a = 0
while a<3:
    print('\n')
    a+=1

print("==========Задание№5============")
T, Y = 10000, 10000
F = np.random.randint(0, 100, size=(T, Y))
start_time5 = time.time()
h = 76
pod_mat = F[0:1000, 0:1000]
result = pod_mat * h
end_time5 = time.time()
print(f"Подматрица(flattened) помноженная на 76:{result.flatten()}")
print(f"Рантайм:{(end_time5 - start_time5):.4f} сек")