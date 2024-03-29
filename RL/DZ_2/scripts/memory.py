
# Библиотека numpy
import numpy as np
# Импортируем модуль для генерации рандомных значений
import random

def add_to_memory(memory,
                  previous_state,
                  action,
                  reward,
                  current_state,
                  episode_done):

  ''' Функция записи информации в память

      Args:
        previous_state — массивы из состояния среды
        action — действие, которое было в нем принято
        reward — награда, которая была получена
        current_state — следующее состояние, к которому действие привело
        episode_done — булевое значение флагов окончания игры (кадр последний в эпизоде)

      Returns:
  '''

  # memory — глобальная переменная. Мы записываем в нее всю нужную информацию:
  memory.append((previous_state, action, reward, current_state, episode_done))
  return memory


def sample_from_memory_m(memory,
                         timesteps_per_train,
                         num_frames,
                         image_width,
                         image_height,
                         chanels,
                         batch_size,
                         part_memory = True):

  ''' Функция сэмплирования данных

      Args:

      Returns: распакованные данные
  '''

  if part_memory == True:
      # определим размер памяти
      memory_batch_size = min(batch_size * timesteps_per_train, len(memory))

  else:  memory_batch_size = len(memory)

  # Сэмплим данные
  mini_batch = random.sample(memory, memory_batch_size)
  # Создаем массив из нулей с размерностью предыдущих состояний, массива действий, массива наград, текущих состояний, флагов окончания игры
  previous_states = np.zeros((memory_batch_size, num_frames, image_width, image_height, chanels))
  actions = np.zeros(memory_batch_size)
  rewards = np.zeros(memory_batch_size)
  current_states = np.zeros((memory_batch_size, num_frames, image_width, image_height, chanels))
  episode_done = np.zeros(memory_batch_size)

  # Перебираем данные и копируем их значения в массивы нулей.
  for i in range(memory_batch_size):
    previous_states[i, :, :, :, :] = mini_batch[i][0]
    actions[i] = mini_batch[i][1]
    rewards[i] = mini_batch[i][2]
    current_states[i, :, :, :, :] = mini_batch[i][3]
    episode_done[i] = mini_batch[i][4]

  return previous_states, actions.astype(np.uint8), rewards, current_states, episode_done