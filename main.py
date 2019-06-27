from model import StyleTransferModel
from neural_style.transformer_net import TransformerNet

import telebot
from telegram_token import token
from config import ProxyURL, StartMsg, WantTalkMsg1, WantTalkMsg2, CancelMsg, WaitStylingMsg, WaitStylingMsg10min, \
    NextActMsg
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

CANDY, MOSAIC, RAIN, UDNIE, OWN, NEXT_PHOTO, WANT_TALK, NEXT_ACT = range(8)
reply_keyboard = [['candy', 'mosaic', 'rain', 'udnie'], ['OWN STYLE', 'I do not want to continue']]


# реакция на "/start"
def start(bot, update):
    print('User Start')
    update.message.reply_text(
        StartMsg,
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True))

    return WANT_TALK


# метод  - реакция на нажатие кнопок Yes/No"
def want_talk(bot, update):
    text = update.message.text
    print(text)
    if text == reply_keyboard[0][0]:
        update.message.reply_text(WantTalkMsg1, reply_markup=ReplyKeyboardRemove())
        return CANDY
    elif text == reply_keyboard[0][1]:
        update.message.reply_text(WantTalkMsg1, reply_markup=ReplyKeyboardRemove())
        return MOSAIC
    elif text == reply_keyboard[0][2]:
        update.message.reply_text(WantTalkMsg1, reply_markup=ReplyKeyboardRemove())
        return RAIN
    elif text == reply_keyboard[0][3]:
        update.message.reply_text(WantTalkMsg1, reply_markup=ReplyKeyboardRemove())
        return UDNIE
    elif text == reply_keyboard[1][0]:
        update.message.reply_text(WantTalkMsg2, reply_markup=ReplyKeyboardRemove())
        return OWN
    else:
        return cancel(bot, update)


# реакция на "/cancel"
def cancel(bot, update):
    print('User Cancel')
    update.message.reply_text(CancelMsg, reply_markup=ReplyKeyboardRemove())

    return ConversationHandler.END


def error(bot, update, error):
    # Log Errors caused by Updates.
    logger.warning('Update "%s" caused error "%s"' % (update, error))


# Получаем две картинки, после второй запускаем перенос стиля (transfer_style)
def send_prediction_on_photo_own(bot, update):
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))

    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)

    if chat_id in first_image_file:
        print('    -the second (style) image')
        # первая картинка, которая к нам пришла станет content image, а вторая style image
        update.message.reply_text(WaitStylingMsg10min)
        content_image_stream = BytesIO()
        first_image_file[chat_id].download(out=content_image_stream)
        del first_image_file[chat_id]

        style_image_stream = BytesIO()
        image_file.download(out=style_image_stream)
        style_type = 'own'
        output = model.transfer_style(content_image_stream, style_image_stream, style_type)

        # теперь отправим назад фото
        output_stream = BytesIO()
        output.save(output_stream, format='PNG')
        output_stream.seek(0)
        bot.send_photo(chat_id, photo=output_stream)
        print("Sent Photo to user")

        update.message.reply_text(
            NextActMsg,
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True))
        return WANT_TALK
        # update.message.reply_text(AfterStylingMsg)

    else:
        print('    -the first (content) image')
        first_image_file[chat_id] = image_file
        return NEXT_PHOTO


def send_prediction_on_photo_candy(bot, update):
    update.message.reply_text(WaitStylingMsg)
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))
    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)
    print('    -the content image')
    content_image_stream = BytesIO()
    image_file.download(out=content_image_stream)

    style_type = 'candy'
    output = model.transfer_style(content_image_stream, content_image_stream, style_type)
    # теперь отправим назад фото
    output_stream = BytesIO()
    output.save(output_stream, format='PNG')
    output_stream.seek(0)
    bot.send_photo(chat_id, photo=output_stream)
    print("Sent Photo to user")
    update.message.reply_text(
        NextActMsg,
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True))
    return WANT_TALK


def send_prediction_on_photo_mosaic(bot, update):
    update.message.reply_text(WaitStylingMsg)
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))
    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)
    print('    -the content image')
    content_image_stream = BytesIO()
    image_file.download(out=content_image_stream)
    style_type = 'mosaic'
    output = model.transfer_style(content_image_stream, content_image_stream, style_type)
    # теперь отправим назад фото
    output_stream = BytesIO()
    output.save(output_stream, format='PNG')
    output_stream.seek(0)
    bot.send_photo(chat_id, photo=output_stream)
    print("Sent Photo to user")
    update.message.reply_text(
        NextActMsg,
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True))
    return WANT_TALK


def send_prediction_on_photo_rain_princess(bot, update):
    update.message.reply_text(WaitStylingMsg)
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))
    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)
    print('    -the content image')
    content_image_stream = BytesIO()
    image_file.download(out=content_image_stream)
    style_type = 'rain_princess'
    output = model.transfer_style(content_image_stream, content_image_stream, style_type)
    # теперь отправим назад фото
    output_stream = BytesIO()
    output.save(output_stream, format='PNG')
    output_stream.seek(0)
    bot.send_photo(chat_id, photo=output_stream)
    print("Sent Photo to user")
    update.message.reply_text(
        NextActMsg,
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True))
    return WANT_TALK


def send_prediction_on_photo_udnie(bot, update):
    update.message.reply_text(WaitStylingMsg)
    chat_id = update.message.chat_id
    print("Got image from {}".format(chat_id))
    # получаем информацию о картинке
    image_info = update.message.photo[-1]
    image_file = bot.get_file(image_info)
    print('    -the content image')
    content_image_stream = BytesIO()
    image_file.download(out=content_image_stream)
    style_type = 'udnie'
    output = model.transfer_style(content_image_stream, content_image_stream, style_type)
    # теперь отправим назад фото
    output_stream = BytesIO()
    output.save(output_stream, format='PNG')
    output_stream.seek(0)
    bot.send_photo(chat_id, photo=output_stream)
    print("Sent Photo to user")
    update.message.reply_text(
        NextActMsg,
        reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True, resize_keyboard=True))
    return WANT_TALK


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
            WANT_TALK: [RegexHandler('^(candy|mosaic|rain|udnie|OWN STYLE|I do not want to continue)$', want_talk)],

            CANDY: [MessageHandler(Filters.photo, send_prediction_on_photo_candy)],
            MOSAIC: [MessageHandler(Filters.photo, send_prediction_on_photo_mosaic)],
            RAIN: [MessageHandler(Filters.photo, send_prediction_on_photo_rain_princess)],
            UDNIE: [MessageHandler(Filters.photo, send_prediction_on_photo_udnie)],
            OWN: [MessageHandler(Filters.photo, send_prediction_on_photo_own)],

            NEXT_PHOTO: [MessageHandler(Filters.photo, send_prediction_on_photo_own)]
        },

        fallbacks=[CommandHandler('cancel', cancel)]
    )

    dp.add_handler(conv_handler)

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()
    # Останавливаем бота, если были нажаты Ctrl + C
    # updater.idle()
