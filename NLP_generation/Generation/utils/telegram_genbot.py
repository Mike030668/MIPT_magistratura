import gdown  # библиотека по работе с файлами в том числе и с гугл_диска
import pandas as pd
import numpy as np
import random
import torch
import gc
from transformers import AutoTokenizer, AutoModel
import sys

from peft import get_peft_model, PeftConfig, PeftModel, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, GenerationConfig

import telebot
from telebot import types

import warnings
warnings.filterwarnings('ignore')


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAIN_DIR =  "/content/MIPT_magistratura/NLP_generation/Generation"

sys.path.append(MAIN_DIR)
from utils.download import gd_folder_download
from utils.constant import SHARE_DIR, TOKEN
from utils.generate import generate_answer, get_prompt, ROLES


# Подгружаем с публичной папки по https
dir_to = MAIN_DIR + "/TinyLamma/"

def main():

    print('Please wait messsage - READY\n')
    print('Load folder with weights and model and tokenizer\n')
    # код потребует гугл авторизацию любую из имеющихся
    model_dir = gd_folder_download(SHARE_DIR, dir_to)

    config = PeftConfig.from_pretrained(model_dir)
    # Loading the model with double quantization
    model_name = "PY007/TinyLlama-1.1B-step-50K-105b"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map='auto'
        )

    # Creating tokenizer and defining the pad token
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              trust_remote_code=True, 
                                              padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token


    # Creating tokenizer and definin
    print('Model loaded\n')


    # Создаем экземпляр бота
    bot = telebot.TeleBot(TOKEN)

    # Функция, обрабатывающая команду /start
    @bot.message_handler(commands=["start"])
    def start(m, res=False):
        # Добавляем две кнопки
        markup=types.ReplyKeyboardMarkup(resize_keyboard=True)
        for role in ROLES:
            item=types.KeyboardButton(role)
            markup.add(item)

        bot.send_message(m.chat.id, 'Бот запущен. Начните общение')

    conext_memory = ''
    get_role = False
    # Получение сообщений от пользователя
    @bot.message_handler(content_types=["text"])
    def handle_text(message):
        global conext_memory, get_role

        #print("in_conext_memory", conext_memory)
        #print()

        if not get_role:
            for role in ROLES:
                #print("message.text", message.text)
                #print(role)
                if role in message.text:
                    #print("get_role ", role)
                    get_role = role
                    break

            if not get_role: get_role = "Kyle"

        if get_role:
            prompt = get_prompt(query = message.text, context = conext_memory, role = get_role)
            #print("have role" , get_role)


        gen_answer = generate_answer(prompt, model, tokenizer, max_new_tokens = 250, device = DEVICE)

        conext_memory = '.'.join([conext_memory, message.text, gen_answer])

        #print("gen_answer", gen_answer)
        #print()
        #print("out_conext_memory", conext_memory)
        # Отсылаем сообщение в чат пользователя
        try:
          bot.send_message(message.chat.id, gen_answer)
        except: pass

    # Запускаем бота
    print('READY  to talk in - https://t.me/SouthPark_test_bot\n')
    bot.polling(none_stop=True, interval=0)


if __name__ == "__main__":
    main()