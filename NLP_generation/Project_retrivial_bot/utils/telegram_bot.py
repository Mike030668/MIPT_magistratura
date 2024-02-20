import gdown  # библиотека по работе с файлами в том числе и с гугл_диска
import pandas as pd
import numpy as np
import random
import torch
import gc
from transformers import AutoTokenizer, AutoModel
import sys
import telebot

import warnings
warnings.filterwarnings('ignore')



MAIN_DIR =  "/content/MIPT_magistratura/NLP_generation/Project_retrivial_bot"
sys.path.append(MAIN_DIR)
from utils import get_replies, load_weights, CrossEncoderBert
from talk_context import flush_memory, get_best_rand_reply

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 128

def main():

    print('Please wait messsage - READY\n')

    # load data
    path_df = MAIN_DIR + "/data/talks_dataset.df"
    talks_df = pd.read_pickle(path_df)
    all_replies = get_replies(talks_df)

    print('Load weights and model\n')
    url = 'https://drive.google.com/file/d/1yj0VuOJmpwio0h9cLuBs50oG_YzVr86a/view?usp=sharing'
    weights =  '/model.pt'
    load_weights(url_weights = url, name_model = "cross_encoder", name_weights = weights, main_dir = MAIN_DIR)

    cross_model = torch.load(MAIN_DIR +"/models/cross_encoder/model.pt")
    cross_model.to(DEVICE)
    print('Model loaded\n')

    token = "6719071574:AAFMsDgoAwJx6C1XiIn0Yb51OQhKMRE2fEg"

    # Создаем экземпляр бота
    bot = telebot.TeleBot(token)


    # Функция, обрабатывающая команду /start
    @bot.message_handler(commands=["start"])
    def start(m, res=False):
        bot.send_message(m.chat.id, 'Бот запущен. Начните общение с ним.')

    #global conext_memory
    #conext_memory = ''
    # Получение сообщений от пользователя
    @bot.message_handler(content_types=["text"])
    def handle_text(message):

        try: CONTEXT_MEMORY
        except: CONTEXT_MEMORY = ''
        #print(CONTEXT_MEMORY)
      
        best_answer, CONTEXT_MEMORY, _ = get_best_rand_reply(
                                cross_model,
                                query = message.text,
                                context = CONTEXT_MEMORY,
                                corpus = all_replies,
                                max_length=MAX_LENGTH,
                                device = DEVICE
                                )

        bot.send_message(message.chat.id, best_answer)

    # Запускаем бота
    print('READY\n')
    bot.polling(none_stop=True, interval=0)


if __name__ == "__main__":
    main()