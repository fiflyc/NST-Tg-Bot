from aiogram import Bot, Dispatcher, executor
from aiogram.types import Message
from aiogram.types.file import File
from aiogram.types.message import ContentType as mctp
from aiogram.dispatcher import filters
from nst_tg_bot.text import Text
import nst_tg_bot.config as config
from nst_tg_bot.file_manager import FileManager
from nst_tg_bot.request_handler import RequestHandler, InputType


bot = Bot(token=config.TOKEN)
dp = Dispatcher(bot)
fmanager = FileManager(bot,
	                   config.DOWNLOADS,
	                   config.CLEARING_TIME,
	                   config.DATABASES)
handler = RequestHandler(fmanager,
                         config.DATABASES,
                         config.WAITING_TIME)


@dp.message_handler(filters.Command(['start'], ignore_caption=False),
                    content_types=mctp.TEXT)
async def on_start_cmd(message: Message):
	await message.reply(Text.START_MESSAGE)

@dp.message_handler(filters.Command(['help'], ignore_caption=False),
                    content_types=mctp.TEXT)
async def on_help_cmd(message: Message):
	await message.reply(Text.HELP_MESSAGE)

@dp.message_handler(filters.Command(['info'], ignore_caption=False),
                    content_types=mctp.TEXT)
async def on_info_cmd(message: Message):
	await message.reply(message)

@dp.message_handler(filters.Command(['links'], ignore_caption=False),
                    content_types=mctp.TEXT)
async def on_links_cmd(message: Message):
	await message.reply(Text.LINKS_MESSAGE)

async def save_content_and_execute(file: File, message: Message):
	chat_id = message.chat['id']
	await handler.set_input(InputType.CONTENT, file, chat_id)
	result = handler.execute_query(chat_id)

	if result is None:
		await message.reply(Text.SEND_STYLE)
	else:
		await bot.send_photo(photo=open(result.name, 'rb'),
		                     chat_id=chat_id,
		                     reply_to_message_id=message.message_id)
		result.close()

@dp.message_handler(filters.Command(['content'], ignore_caption=False),
                    content_types=mctp.PHOTO)
async def on_content_image(message: Message):
	file = await bot.get_file(message.photo[-1]['file_id'])
	await save_content_and_execute(file, message)

@dp.message_handler(filters.Command(['content'], ignore_caption=False),
                    content_types=mctp.DOCUMENT)
async def on_content_file(message: Message):
	if 'image' not in message.document['mime_type']:
		await message.reply(Text.WRONG_FILE_FORMAT)
	else:
		file = await bot.get_file(message.document['file_id'])
		await save_content_and_execute(file, message)

@dp.message_handler(filters.Command(['content']),
                    content_types=mctp.TEXT)
async def on_forwarded_content(message: Message):
	if message.reply_to_message is None:
		await message.reply(Text.NO_CONTENT_IMAGE)
	elif not message.reply_to_message.photo and \
	     not message.reply_to_message.document:
		await message.reply(Text.NO_CONTENT_IMAGE)
	elif not message.reply_to_message.photo:
		if 'image' not in message.reply_to_message.document['mime_type']:
			await message.reply(Text.WRONG_FILE_FORMAT)
		else:
			file = await bot.get_file(message.reply_to_message.document['file_id'])
			await save_content_and_execute(file, message)
	else:
		file = await bot.get_file(message.reply_to_message.photo[-1]['file_id'])
		await save_content_and_execute(file, message)

async def save_style_and_execute(file: File, message: Message):
	chat_id = message.chat['id']
	await handler.set_input(InputType.STYLE, file, chat_id)
	result = handler.execute_query(chat_id)

	if result is None:
		await message.reply(Text.SEND_CONTENT)
	else:
		await bot.send_photo(photo=open(result.name, 'rb'),
		                     chat_id=chat_id,
		                     reply_to_message_id=message.message_id)
		result.close()

@dp.message_handler(filters.Command(['style'], ignore_caption=False),
                    content_types=mctp.PHOTO)
async def on_style_image(message: Message):
	file = await bot.get_file(message.photo[-1]['file_id'])
	await save_style_and_execute(file, message)

@dp.message_handler(filters.Command(['style'], ignore_caption=False),
                    content_types=mctp.DOCUMENT)
async def on_style_file(message: Message):
	if 'image' not in message.document['mime_type']:
		await message.reply(Text.WRONG_FILE_FORMAT)
	else:
		file = await bot.get_file(message.document['file_id'])
		await save_style_and_execute(file, message)

@dp.message_handler(filters.Command(['style']),
                    content_types=mctp.TEXT)
async def on_forwarded_style_image(message: Message):
	if message.reply_to_message is None:
		await message.reply(Text.NO_STYLE_IMAGE)
	elif not message.reply_to_message.photo and \
	     not message.reply_to_message.document:
		await message.reply(Text.NO_STYLE_IMAGE)
	elif not message.reply_to_message.photo:
		if 'image' not in message.reply_to_message.document['mime_type']:
			await message.reply(Text.WRONG_FILE_FORMAT)
		else:
			file = await bot.get_file(message.reply_to_message.document['file_id'])
			await save_style_and_execute(file, message)
	else:
		file = await bot.get_file(message.reply_to_message.photo[-1]['file_id'])
		await save_style_and_execute(file, message)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
