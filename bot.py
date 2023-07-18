import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import pickle
from collections import defaultdict
import random
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

import warnings
warnings.filterwarnings("ignore")

TOKEN = '6074844285:AAHN-_TYEwklIM_mYtN4n8UjryP9zS_P350'
bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)

# Загрузка модели и токенизатора
model_name = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

# Загрузка датасета с вещами tsum.ru
clothes = pd.read_csv('all+++.csv')  # пока тут другой датасет
clothes['author'].fillna('other', inplace=True)
categories = ['up', 'bottom', 'full', 'shoes', 'bag', 'acc', 'outwear']
clothes['random_category'] = random.choices(categories, k=len(clothes))

length = 256

with open("embeddings_256.pkl", "rb") as f:
    embeddings = pickle.load(f)


def generate_recommendations(query, embeddings):
    top_elements_per_category = defaultdict(list)

    for category in categories:
        query_tokens = tokenizer.encode_plus(
            query,
            add_special_tokens=True,
            max_length=length,
            pad_to_max_length=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            query_outputs = model(**query_tokens)
            query_hidden_states = query_outputs.hidden_states
            query_last_hidden_state = query_hidden_states[-1][:, 0, :]
            query_embedding = query_last_hidden_state.squeeze()
            cosine_similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                torch.stack(embeddings)
            )

            cosine_similarities = cosine_similarities.numpy()

            indices = np.argsort(cosine_similarities)[::-1]

            elements = []
            for i in indices:
                if clothes['random_category'][i] == category:
                    url = clothes['page_url'][i]
                    probability = cosine_similarities[i]
                    elements.append((url, probability))
                    if len(elements) == 2:
                        break

            top_elements_per_category[category] = elements

    return top_elements_per_category


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.reply("Привет! Какой образ вы хотели бы подобрать?")


@dp.message_handler()
async def process_outfit(message: types.Message):
    outfit = message.text.strip()

    await message.reply("Хороший выбор! В каком стиле вы хотели бы подобрать образ?",
                        reply_markup=ReplyKeyboardMarkup(resize_keyboard=True, selective=True).add(
                            KeyboardButton(text='Классика (Old Money)'),
                            KeyboardButton(text='Женственный (Femininity)'),
                            KeyboardButton(text='Авангардный (Avant-garde)'),
                            KeyboardButton(text='Минимализм (Minimalism)'),
                            KeyboardButton(text='Бохо (Bohemian)'),
                            KeyboardButton(text='Эклектичный (Eclectic)')
                        ))

    await types.ChatActions.typing()

    # Передаем переменную outfit в функцию process_style
    dp.register_message_handler(process_style, lambda message: True, state='*', outfit=outfit)


async def process_style(message: types.Message, state, outfit):
    chosen_style = message.text.strip()
    query = f"{outfit} в стиле {chosen_style}"
    top_elements_per_category = generate_recommendations(query, embeddings)

    for category, elements in top_elements_per_category.items():
        await message.reply(f'{category}:')
        await message.reply(f'{len(elements)} элементов')
        for url, probability in elements:
            await message.reply(f'{url}: {probability:.4f}')

    await message.reply("Какой образ вы хотели бы подобрать?")


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
