from model import StyleTransferModel
from neural_style.transformer_net import TransformerNet

import telebot
from telegram_token import token
from config import ProxyURL, StartMsg, WantTalkMsg, CancelMsg, SndPict2Msg, WaitStylingMsg, AfterStylingMsg
import numpy as np
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms

import re

from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters, RegexHandler, ConversationHandler)
import logging

"""
Send /start to initiate the conversation.
"""

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

model = StyleTransferModel()
first_image_file = {}

WANT_TALK, PHOTO, NEXT_PHOTO = range(3)
reply_keyboard = [['YES', 'NO']]

# реакция на "/start"
def start(bot, update):

    print('User Start')
    update.message.reply_text(
        StartMsg,
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True))

    return WANT_TALK
# метод  - реакция на нажатие кнопок Yes/No"
def want_talk(bot, update, user_data):

     text = update.message.text
     if text == reply_keyboard [0][0]:
         update.message.reply_text(WantTalkMsg, reply_markup=ReplyKeyboardRemove())

         return PHOTO
     else:
         return cancel(bot, update)

# реакция на "/cancel"
def cancel(bot, update):
    print('User Cancel')
    update.message.reply_text(CancelMsg, reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END

def error(bot, update, error):
    #Log Errors caused by Updates.
    logger.warning('Update "%s" caused error "%s"' % (update, error))

# Получаем две картинки, после второй запускаем перенос стиля (transfer_style)
def send_prediction_on_photo(bot, update):
    # Нам нужно получить две картинки, чтобы произвести перенос стиля, но каждая картинка приходит в
    # отдельном апдейте, поэтому в простейшем случае мы будем сохранять id первой картинки в память,
    # чтобы, когда уже придет вторая, мы могли загрузить в память уже сами картинки и обработать их.

    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))

    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)

    if chat_id in first_image_file:
        print('    -the second (style) image')
        # первая картинка, которая к нам пришла станет content image, а вторая style image
        update.message.reply_text(WaitStylingMsg)
        content_image_stream = BytesIO()
        first_image_file[chat_id].download(out=content_image_stream)
        del first_image_file[chat_id]

        style_image_stream = BytesIO()
        image_file.download(out=style_image_stream)


        output = model.transfer_style(content_image_stream, style_image_stream)

        # теперь отправим назад фото
        output_stream = BytesIO()
        output.save(output_stream, format='PNG')
        output_stream.seek(0)
        bot.send_photo(chat_id, photo=output_stream)
        print("Sent Photo to user")
        update.message.reply_text(AfterStylingMsg)
    else:
        print('    -the first (content) image')
        first_image_file[chat_id] = image_file
        update.message.reply_text(SndPict2Msg)
        return NEXT_PHOTO

if __name__ == '__main__':

    # используем прокси"socks4 proxy"

    # создаём апдейтер и передаём им наш токен, который был выдан после создания бота
    updater = Updater(token=token, request_kwargs={'proxy_url': ProxyURL})
    # определяем диспетчер для регистрации обработчиков
    dp = updater.dispatcher

    # инициируем обработчики для диалога
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={
            WANT_TALK: [RegexHandler('^(YES|NO)$', want_talk, pass_user_data=True)],

            PHOTO: [MessageHandler(Filters.photo, send_prediction_on_photo)],
            NEXT_PHOTO: [MessageHandler(Filters.photo, send_prediction_on_photo)]
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )


    dp.add_handler(conv_handler)

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()
    # Останавливаем бота, если были нажаты Ctrl + C
    #updater.idle()
