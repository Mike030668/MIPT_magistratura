import gdown  # библиотека по работе с файлами в том числе и с гугл_диска
from transformers import AutoTokenizer, AutoModel
import torch
import random

MAX_LENGTH = 128

def get_replies(df):
    speakers = ['Mr. Adler', 'Mr. Garrison', 'Mr. Hankey', 'Mr. Hat', 'Mr. Mackey', 'Mr. Slave',
            'Mrs. Garrison', 'Mrs. McCormick', 'Ms. Choksondik', 'Ms. Crabtree', 'Ms. Ellen', 
              'Jimbo', 'Jimmy', 'Joe', 'Josh', 'Kenny', 'Kids', 'Kyle',
            'Kyle Two', 'Larry', 'Liane', 'Linda', 'Manager', 'Mark', 'Marvin', 'Mayor',  
            'Dr. Chinstrap', 'Dr. Doctor', 'Gary', 'Tom Cruise', 'Towelie', 'Tuong Lu Kim', 'Tweek', 'Wendy']
    # Соберем все ответы из базы
    base_answers = df[df["speaker"].isin(speakers)]['close_reply'].values
    replies = []
    for rep in base_answers.tolist()[:-100]:
        if len(rep): rep = random.choice(rep)
        else: pass
        if 0< len(rep)<100 and "?" not in rep: replies.append(rep)
    return list(set(replies)) # Список всех ответов из базы


def load_weights(name_model, name_weights, url_weights, main_dir):
    dir_model = main_dir + f"/models/{name_model}"

    # левая часть адреса
    main = 'https://drive.google.com/uc?id='
    # Подгружаем на диск ноута файл индексации
    index_id = url_weights.split('/')[-2]

    # Подгружаем на диск ноута файл индексации
    gdown.download(main + index_id, dir_model + name_weights, quiet=False)


class CrossEncoderBert(torch.nn.Module):
    def __init__(self, max_length: int = MAX_LENGTH):
        super().__init__()
        self.max_length = max_length
        self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.bert_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        self.linear = torch.nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use the CLS token's output
        return self.linear(pooled_output)

