from tensorflow.keras.models import load_model, Model, Sequential
# Все слои из кераса
from tensorflow.keras.layers import *
# Оптимизатор RMSprop
from tensorflow.keras.optimizers import RMSprop
# Библиотека тензорфлоу
import tensorflow as tf

from memory import sample_from_memory_m

def Make_DQN(input_shape,
             action_size,
             learning_rate):

    model = Sequential()

    model.add(
        TimeDistributed(
            Conv2D(64, (3,3),
                padding='same', strides=(2,2), activation='relu'),
            input_shape = input_shape
        )
    )
    model.add(
        TimeDistributed(
            Conv2D(64, (2,2),
                padding='same', strides=(1,1), activation='relu')
        )
    )
    model.add(
        TimeDistributed(
            MaxPooling2D((2,2), strides=(2,2))
        )
    )
    # Second conv, 128
    model.add(
        TimeDistributed(
            Conv2D(128, (3,3),
                padding='same', strides=(2,2), activation='relu')
        )
    )
    model.add(
        TimeDistributed(
            Conv2D(128, (2,2),
                padding='same', strides=(1,1), activation='relu')
        )
    )
    model.add(
        TimeDistributed(
            MaxPooling2D((2,2), strides=(2,2))
        )
    )

    # Second conv, 128
    model.add(
        TimeDistributed(
            Conv2D(128, (3,3),
                padding='same', strides=(2,2), activation='relu')
        )
    )
    model.add(
        TimeDistributed(
            Conv2D(128, (2,2),
                padding='same', strides=(1,1), activation='relu')
        )
    )
    model.add(
        TimeDistributed(
            MaxPooling2D((2,2), strides=(2,2))
        )
    )

    ## and so on with 512, 1024...
    ## ...
    # then we can use Flatten to reduce dimension to 1
    model.add(TimeDistributed(Flatten()))
    model.add(Conv1D(128, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(64, return_sequences=True))
    model.add(Activation('relu'))
    model.add(LSTM(32))

    model.add(Dense(16, activation = 'relu'))
    ## and then... merge the entire outputs to
    ## be able to use Dense(), and make predictions...
    model.add(Dense(action_size, activation = 'linear'))
    # Практика показывает что RMSprop хороший оптимизатор для обучения с подкреплением, однако можно использовать adam.
    optimizer = RMSprop(learning_rate = learning_rate)

    # Компилируем модель с функции ошибки mse и заданным оптимизатором.
    model.compile(loss = "mse", optimizer = optimizer)

    return model


def update_target_model(target_model,
                        main_model
                        ):

  ''' Функция обновления весов в целевой модели, т.е. той
      Устанавливает веса целевой модели (которая не обучается) такими
      же как веса основной модели (которая обучается)

  '''
  target_model.set_weights(main_model.get_weights())


import gc
# Custom Callback To Include in Callbacks List At Training Time
class GarbageCollectorCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

clear_ozu = GarbageCollectorCallback()



def train_network(
                  main_model,
                  target_model,
                  memory,
                  part_memory,
                  batch_size,
                  gamma,

                  ):

  ''' Функция обучения алгоритма

      Args:

      Returns: обученная модель
  '''

  # Извлекаем пакет данных из памяти
  previous_states, actions, rewards, current_states, game_finished = sample_from_memory_m(memory, part_memory)

  # Предсказываем Q(s, a)
  Q_values = main_model.predict(previous_states)

  # Предсказываем Q(s', a')
  next_Q_values = target_model.predict(current_states)
  # Модифицируем значения Q
  for i in range(current_states.shape[0]):
    # Если состоянее последнее в эпизоде
    if game_finished[i]:
        Q_values[i, actions[i]] = rewards[i]
    # Если состояние не последнее в эпизоде
    else:
        Q_values[i, actions[i]] = rewards[i] + gamma * next_Q_values[i, actions[i]]

  # dataset
  dataset = tf.data.Dataset.from_tensor_slices((previous_states, Q_values))
  dataset = dataset.batch(batch_size)

  # Обучаем модель
  main_model.fit(dataset, verbose = 0, callbacks=[clear_ozu])