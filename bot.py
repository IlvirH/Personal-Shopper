import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import re
import pickle
import requests
from io import BytesIO
from telegram.ext import Updater, CommandHandler, MessageHandler, filters
from telegram import Update, ForceReply

# Загрузка модели и токенизатора
model_name = "cointegrated/rubert-tiny2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

# Загрузка датасета и аннотаций к книгам
books = pd.read_csv('all+++.csv')
books['author'].fillna('other', inplace=True)

annot = books['annotation']

# Получение эмбеддингов аннотаций каждой книги в датасете
length = 256

with open("embeddings_256.pkl", "rb") as f:
    book_embeddings = pickle.load(f)

def generate_recommendations(query):
    query_tokens = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=length,
        pad_to_max_length=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        query_outputs = model(**query_tokens)
        query_hidden_states = query_outputs.hidden_states[-1][:, 0, :]
        query_hidden_states = torch.nn.functional.normalize(query_hidden_states)

    cosine_similarities = torch.nn.functional.cosine_similarity(
        query_hidden_states.squeeze(0),
        torch.stack(book_embeddings)
    )

    cosine_similarities = cosine_similarities.numpy()

    indices = np.argsort(cosine_similarities)[::-1]

    return indices

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Привет! Введите запрос для получения книжных рекомендаций.")

def process_message(update, context):
    query = update.message.text.strip()
    indices = generate_recommendations(query)

    num_books_per_page = 3  # Количество книг на странице

    for i in indices[:num_books_per_page]:
        context.bot.send_message(chat_id=update.effective_chat.id, text="## " + books['title'][i])
        context.bot.send_message(chat_id=update.effective_chat.id, text="**Автор:** " + books['author'][i])
        context.bot.send_message(chat_id=update.effective_chat.id, text="**Аннотация:** " + books['annotation'][i])
        image_url = books['image_url'][i]
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        context.bot.send_photo(chat_id=update.effective_chat.id, photo=image)
        context.bot.send_message(chat_id=update.effective_chat.id, text=f"**Cosine similarity:** {cosine_similarities[i]}.")

def main():
    # Создание объекта Updater и передача токена вашего бота, а также update_queue
    updater = Updater("6074844285:AAHN-_TYEwklIM_mYtN4n8UjryP9zS_P350", update_queue=True)

    # Добавление обработчика команды /start
    updater.dispatcher.add_handler(CommandHandler("start", start))

    # Добавление обработчика сообщений
    updater.dispatcher.add_handler(MessageHandler(filters.text, process_message))

    # Запуск бота
    updater.start_polling()

    # Остановка бота при нажатии Ctrl+C
    updater.idle()

if __name__ == '__main__':
    main()

