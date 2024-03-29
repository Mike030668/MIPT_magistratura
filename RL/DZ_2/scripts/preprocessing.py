import cv2
import numpy as np

def preprocess_frame(frame,
                     image_width,
                     image_height,
                     chanels):

  ''' Функция преобразования изображений

      Args:
        frame -

      Returns:
        Возвращаем предобработанное, нормализованное, решейпнутое изображение

  '''
  # Меняем оси
  frame = np.rollaxis(frame, 0, 3)

  # Меняем размерность картинки на (64 х 64)
  frame = cv2.resize(frame, (image_width, image_height), interpolation=cv2.INTER_CUBIC)

  if chanels == 1:
      # Переводим в чёрно-белым
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      norm_frame = frame.reshape(image_width, image_height, 1)/255
      return norm_frame

  elif chanels == 3:
      img_stack_sm = np.zeros((image_width, image_height, chanels))

      for idx in range(chanels):
          img = frame[:, :, idx]
          img_sm = cv2.resize(img, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
          img_stack_sm[:, :, idx] = img_sm
      norm_frame = img_stack_sm/255
      return norm_frame
  
