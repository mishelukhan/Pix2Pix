import logging
import os
import random
import image_processing
import load_models
from networks import Generator
from config import API_TOKEN, PATH_GENERATOR, CACHE_SHOES, CACHE_EDGES

from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import ReplyKeyboardRemove, ReplyKeyboardMarkup, KeyboardButton
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.types.input_file import InputFile
from aiogram.utils.markdown import link
import emoji

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await types.ChatActions.typing()
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True).add(types.KeyboardButton(text="Guide"))
    keyboard.add(types.KeyboardButton(text="Example"))
    keyboard.add(types.KeyboardButton(text="Load image"))
    keyboard.add(types.KeyboardButton(text="FAQ"))
    message_welcome = (
        f"Hi, {message.from_user.first_name} {emoji.emojize(':love-you_gesture:')}!\n"
        f"I'm edges2shoesBot {emoji.emojize(':running_shoe:')}!\n"
        f"You can load edges of shoes {emoji.emojize(':paintbrush:')}"
        f"and I will make real shoes {emoji.emojize(':star-struck:')} from this sketch!\n"
        f"My functionality {emoji.emojize(':toolbox:')} is as follows:\n"
        f"*Guide* {emoji.emojize(':spiral_notepad:')}:\n"
        f"Click on the Guide button and you will see how"
        f" to create a real masterpiece {emoji.emojize(':framed_picture:')} in a few minutes.\n"
        f"*Example* {emoji.emojize(':TOP_arrow:')}: \n"
        f"If you don't know where to start {emoji.emojize(':thinking_face:')}"
        f"just look at the best examples. \n"
        f"*Load image* {emoji.emojize(':counterclockwise_arrows_button:')}: \n"
        f"Upload the shoe edges and you will get a colored {emoji.emojize(':rainbow:')} version. \n"
        f"*FAQ* {emoji.emojize(':red_question_mark:')}:\n"
        f"You can visit repo with code on Github {emoji.emojize(':link:')}" 
        f" to obtain more details.\n"
    )
    await message.reply(message_welcome, reply_markup=keyboard, parse_mode="Markdown")


@dp.message_handler()
async def tasker(message: types.Message):
    if message.text == 'FAQ':
        url_repo = InlineKeyboardMarkup(row_width=2)
        RepoButton = InlineKeyboardButton(text='Github repo', 
                                          url='https://github.com/mishelukhan/pix2pix')
        url_repo.add(RepoButton)
        await message.answer('If you want to know more, please visit our page with code on Github.',
                             reply_markup=url_repo)
    elif message.text == 'Example':
        examples_path = "/mnt/c/projects/pix2pix/examples"
        rand_num = str(random.randint(0, 200))
        test_path = os.path.join(examples_path, 'test' + rand_num)
        edges_path = os.path.join(test_path, 'img_A.png')
        shoes_path = os.path.join(test_path, 'img_fake.png')
        edges = InputFile(edges_path)
        shoes = InputFile(shoes_path)
        await message.answer("You can load contours like this one:")
        await bot.send_photo(chat_id=message.chat.id, photo=edges)
        await message.answer("And I will process it into this one:")
        await bot.send_photo(chat_id=message.chat.id, photo=shoes)
    elif message.text == 'Load image':
        await message.answer('Add 1 picture of shoe contours.')
    elif message.text == 'Guide':
        message_guide = (
            f"First of all, you need to draw the outline "
            f"of the shoes in some drawing application. \n"
            f"To get the best result, it is strongly recommended "
            f"to use the *{link('Pixilart', 'https://www.pixilart.com/draw?ref=home-page')}* "
            f"web application in the following configuration: \n"
            f"{emoji.emojize(':star:')}: Use 256 by 256 pixels resolution.\n"
            f"{emoji.emojize(':star:')}: Use a black pencil with width of one pixel.\n"
            f"{emoji.emojize(':star:')}: Try to draw shoes as big as possible. \n"
            f"{emoji.emojize(':star:')}: Save the image in jpeg or png format, "
            f"do not use the PrintScreen. \n"
            f"{emoji.emojize(':star:')}: And most importantly:"
            f"Realistic contour will lead to high-quality results."
            f"Draw as voluminous and detailed shoes as possible!"
        )
        await message.answer(message_guide, parse_mode="Markdown")
    else:
        await message.answer('Try to use another command.')


@dp.message_handler(content_types=["photo"])
async def process_photo(message: types.Message):
    # Ask user to wait about 30 second
    await message.answer('Please, wait about 30 second...')
    # Download photo
    photo_edges = message.photo[-1]
    await photo_edges.download(destination_file=CACHE_EDGES)
    # Deliver it to generator 
    model = load_models.load_models(Generator, PATH_GENERATOR)
    await image_processing.process(model, CACHE_EDGES, CACHE_SHOES)
    photo_shoes = InputFile(CACHE_SHOES)
    await message.answer('Result:')
    await bot.send_photo(chat_id=message.chat.id, photo=photo_shoes)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
