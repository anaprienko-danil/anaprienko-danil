# Лабораторная Работа №2: Структуры данных
## Задание:
Перемножить 2 квадратные матрицы размера 4096x4096 с элементами типа double.
Исходные матрицы генерируются в программе (случайным образом либо по определенной формуле) либо считываются из заранее подготовленного файла.
Оценить сложность алгоритма по формуле c = 2 n3, где n - размерность матрицы.
Оценить производительность в MFlops, p = c/t*10-6, где t - время в секундах работы алгоритма.
Выполнить 3 варианта перемножения и их анализ и сравнение:
1-й вариант перемножения - по формуле из линейной алгебры. 
2-й вариант перемножения - результат работы функции cblas_dgemm из библиотеки BLAS (рекомендуемая реализация из Intel MKL)
3-й вариант перемножения - оптимизированный алгоритм по вашему выбору, написанный вами, производительность должна быть не ниже 30% от 2-го варианта

## Реализация:
### Листинг программы:
``` python
import numpy as np
import time
import scipy.linalg.blas as blas

def print_author_info():
    print("Автор: Анаприенко Даниил Сергеевич")
    print("Группа: АИСа-о24\n")

def generate_random_matrix(n):
    """Генерация случайной матрицы n x n"""
    return np.random.rand(n, n)

def naive_matrix_multiplication(A, B, n):
    """Наивное умножение матриц"""
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

def blas_matrix_multiplication(A, B):
    """Умножение матриц с использованием BLAS (dgemm)"""
    return blas.dgemm(alpha=1.0, a=A, b=B)

def blocked_matrix_multiplication(A, B, n, block_size=64):
    """Блочное умножение матриц"""
    C = np.zeros((n, n))
    for i in range(0, n, block_size):
        for j in range(0, n, block_size):
            for k in range(0, n, block_size):
                # Определяем границы блоков
                i_end = min(i + block_size, n)
                j_end = min(j + block_size, n)
                k_end = min(k + block_size, n)
                
                # Умножаем блоки
                C[i:i_end, j:j_end] += np.dot(A[i:i_end, k:k_end], B[k:k_end, j:j_end])
    return C

def test_performance(A, B, n, func, func_name):
    """Тестирование производительности"""
    start_time = time.time()
    C = func(A, B, n) if func.__code__.co_argcount == 3 else func(A, B)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    c = 2 * n ** 3  # Вычислительная сложность
    mflops = (c / elapsed_time) / 1e6  # Производительность в MFLOPS
    
    print(f"{func_name}:")
    print(f"  Время: {elapsed_time:.3f} сек")
    print(f"  Производительность: {mflops:.2f} MFlops\n")
    return elapsed_time, mflops

def main():
    print_author_info()
    
    n = 4096  # Размер матрицы
    block_size = 64  # Размер блока для блочного алгоритма
    
    print("Генерация матриц...")
    A = generate_random_matrix(n)
    B = generate_random_matrix(n)
    print("Генерация завершена.\n")
    
    # Тестируем разные методы умножения
    print("Тестирование производительности:")
    
    # Наивный алгоритм (комментируем для n=4096, так как очень долго)
    # print("Наивный алгоритм:")
    # naive_time, naive_mflops = test_performance(A, B, n, naive_matrix_multiplication, "Наивный алгоритм")
    
    # BLAS
    blas_time, blas_mflops = test_performance(A, B, n, blas_matrix_multiplication, "BLAS (dgemm)")
    
    # Блочный алгоритм
    blocked_time, blocked_mflops = test_performance(A, B, n, 
                                                  lambda a, b, sz: blocked_matrix_multiplication(a, b, sz, block_size), 
                                                  "Блочный алгоритм")
    
    # Вывод сравнения производительности
    print("\nСравнение производительности:")
    print(f"BLAS быстрее блочного алгоритма в {blocked_time/blas_time:.1f} раз")
    print(f"Производительность блочного алгоритма составляет {blocked_mflops/blas_mflops*100:.1f}% от BLAS")

if __name__ == "__main__":
    main()
```

## Результат выполнения программы:
![image](https://github.com/user-attachments/assets/0e00bbe0-6f9a-47d0-b9dc-bfb2c5f7a735)
