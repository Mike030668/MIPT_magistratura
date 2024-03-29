import numpy as np
import random

def get_action(model,
               state,
               action_size,
               epsilon = 0,
               final_epsilon = 0.01,
               epsilon_decay_factor = 0.99999,
               ):

  ''' Функция предсказания награды за действие

      Args: state -
            epsilon -
            action_size -

      Returns: выбранное действие и новое значение epsilon

  '''

  # Генерим рандомное значение и сравниваем
  if random.random() <= epsilon:
    # дейстие случайное
    type_act = 0
    action_index = np.random.randint(0, action_size)

  # Иначе (если рандомное число больше чем эпсилон)
  else:
    # Предсказываем все Q-значения при следующим действии (Q(s, a) для каждого действия a)
    Q_values = model.predict(np.expand_dims(state, axis = 0))
    # Извлекаем индекс действия который приводит к максимальному значению Q(s, a)
    action_index = np.argmax(Q_values)
    # дейстие сети
    type_act = 1
  # Снижаем значение эпсилон, если оно больше чем final_epsilon, снижаем значение epsilon на epsilon_decay_factor.
  if epsilon > final_epsilon:
    #Снижаем значение эпсилон умножением (это приведёт к экспоненциальному спаду). Убеждаемся, что значение эпсилон не ниже чем final_epsilon.
    epsilon = max(epsilon * epsilon_decay_factor, final_epsilon) 

  return action_index, epsilon, type_act



def get_reward(previous_info,
               current_info,
               episode_done
               ):

    ''' Функция предобработки наград

        Args:
            previous_misc - информация о игровой среде на предидущий кадр (количество убитых врагов, патроны, и здоровие)
            current_misc - информация о игровой среде на текущий кадр (количество убитых врагов, патроны, и здоровие)
            episode_done - булевое значение, которое говорит если кадр является последним в эпизоде.
            misc[0] - количесто убитых врагов, misc[1] - патроны, misc[2] - здоровье

        Returns: подсчитанная награда

    '''

    # Инициализируем награду как 0
    reward = 0

    psi =  0.00001

    # Если кадр является последним в игре, ставим награду как -0.1 и возвращаем её (агент умер)
    if episode_done:
        reward = -0.5 * previous_info[3]/(current_info[3] + psi)
        return reward

    # Если убили врага в кадре, увеличиваем награду на 1
    if current_info[0] > previous_info[0]:
        reward += 1

    # Если потеряли здоровие, уменьшаем на соотношение разницы к прошлому
    if current_info[1] < previous_info[1]:
        reward -= 0.3*(previous_info[1]-current_info[1])/(previous_info[1] + psi)

    # Если использовали патрон, уменьшаем награду на 0.1
    if current_info[2] < previous_info[2]:
        reward -= 0.1

    # Если живы
    if current_info[3] > previous_info[3]:
       reward += 0.1 * previous_info[3]/(current_info[3] + psi)
       
    return reward