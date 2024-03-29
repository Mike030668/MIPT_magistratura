import numpy as np
# Модуль pyplot из бибиотеки matplotlib
import matplotlib.pyplot as plt
# Функция для управления вывода в колаб-ячейках
from google.colab import output

def moving_average(data,
                   width = 10):

  ''' Функция для подсчета скользящего среднего всех значений

      Args:
        data - входной массив,
        width - длина на которое считаем скользящее среднее

      Returns: результат свёртки данных на фильтр из единиц - наше скользящее среднее
  '''

  # Длина свёртки
  width = min(width, len(data))

  # Создадим паддинг для свёртки
  data = np.concatenate([np.repeat(data[0], width), data])

  # Возвращаем результат свёртки
  return (np.convolve(data, np.ones(width), 'valid') / width)[1:]



def show_scores(scores,
                killcount,
                ammo,
                window
                ):

  ''' Функция визуализации результата

      Args:
        scores -
        killcount -
        ammo -

      Returns: график
  '''

  # Удаляем предыдущий вывод ячейки
  output.clear()

  # Создаем два сабплота (в левом будут награды и средние награды, в правом будут количества убитых врагов и патронов)
  fig, axes = plt.subplots(1, 2, figsize = (20, 8)) # Делаем размер графика большим

  # Устанавливаем большой размер полотна
  axes[0].plot(scores, label = "Награда за эпизод")
  # Отрисовываем скользящие средние награды
  axes[0].plot(moving_average(scores, width = window), label = "Скользящее среднее награды")
  # Добавляем лейблы осей
  axes[0].set_xlabel("Итерация", fontsize = 16)
  axes[0].set_ylabel("Награда", fontsize = 16)
  # Добавляем легенду к графику
  axes[0].legend()

  # Отрисовываем количество убитых врагов
  axes[1].plot(killcount, 'red', linestyle = '--', label = f"Количество убитых врагов (сумма за {window} эпизодов)")
  # Отрисовываем количество убитых врагов (скользящее среднее)
  axes[1].plot(moving_average(killcount, width = window), 'black', label = f"Количество убитых врагов (скользящее среднее за {window} итераций)")
  # Отрисовываем количество оставшихся патронов
  axes[1].plot(ammo, 'green', linestyle = '--', label = f"Осталось патронов (сумма за {window} эпизодов)")
  # Отрисовываем количество оставшихся патронов (скользящее среднее)
  axes[1].plot(moving_average(ammo, width = window), 'blue', label = f"Осталось патронов (скользящее среднее за {window} итераций)")
  # Добавляем лейблы осей
  axes[1].set_xlabel("Итерация", fontsize = 16)
  axes[1].set_ylabel("Значение", fontsize = 16)
  # Добавляем легенду к графику
  axes[1].legend()

  # Отображаем график
  plt.show()