import os
import asyncio
from aiogram import Bot
from aiogram.types.file import File
import sqlite3
from sqlite3 import Error as DBError


class FileManager():

	def __init__(self, bot: Bot,
		         cache_dir: str,
		         clearing_time:int,
		         path_to_db: str):
		self.__bot = bot
		self.__path_to_db = path_to_db + '/files.sqlite'
		self.__cache_dir = cache_dir

		if not os.path.exists(self.__path_to_db):
			try:
				connection = sqlite3.connect(self.__path_to_db)

				cursor = connection.cursor()
				cursor.execute(Queries.CREATE_TABLE)
				connection.commit()
				
				connection.close()
			except DBError as e:
				print(f"DB error occured: {e}")
				quit()

		self.__clearing_time = clearing_time

	async def clear_cache_task(self):
		await asyncio.sleep(self.__clearing_time)

		while True:
			unprotected = []

			try:
				connection = sqlite3.connect(self.__path_to_db)

				unprotected = self.__get_unprotected(connection)
				self.__delete_unprotected(connection)

				connection.close()
			except DBError as e:
				print(f"DB error occured: {e}")

			for file in unprotected:
				os.remove(file[0])

			await asyncio.sleep(self.__clearing_time)

	def __get_unprotected(self, connection):
		cursor = connection.cursor()
		cursor.execute(Queries.GET_UNPROTECTED)
		res = cursor.fetchall()

		return res

	def __delete_unprotected(self, connection):
		cursor = connection.cursor()
		cursor.execute(Queries.DELETE_UNPROTECTED)
		connection.commit()		

	async def get_local_path(self, file: File):
		local_path = None
		# unique single file code:
		file_id = file.file_unique_id

		try:
			connection = sqlite3.connect(self.__path_to_db)

			await asyncio.sleep(0.01)

			if not self.__is_loaded(file_id, connection):
				local_path = await self.__download_file(file)
				await asyncio.sleep(0.01)
				self.__create_entry(file_id, local_path, connection)
			else:
				local_path = self.__find_in_cache(file_id, connection)
				await asyncio.sleep(0.01)
				self.__protect_file(local_path, connection)

			connection.close()
		except DBError as e:
			print(f"DB error occured: {e}")

		return local_path

	def __is_loaded(self, file_id, connection):
		cursor = connection.cursor()
		cursor.execute(Queries.GET_FILE_PATH % file_id)

		return len(cursor.fetchall()) > 0

	async def __download_file(self, file: File):
		local_path = f"{self.__cache_dir}/" + \
				     f"{file.file_unique_id}" + \
				     os.path.splitext(file.file_path)[1]

		await self.__bot.download_file(file.file_path, local_path)
		return local_path

	def __create_entry(self, file_id, local_path, connection):
		cursor = connection.cursor()
		cursor.execute(Queries.CREATE_ENTRY % (file_id, local_path))
		connection.commit()

	def __find_in_cache(self, file_id, connection):
		cursor = connection.cursor()
		cursor.execute(Queries.GET_FILE_PATH % file_id)
		return cursor.fetchall()[0][0]

	def __protect_file(self, local_path, connection):
		cursor = connection.cursor()
		cursor.execute(Queries.PROTECT % (local_path))
		connection.commit()	

	def release_file(self, local_path: str):
		try:
			connection = sqlite3.connect(self.__path_to_db)

			cursor = connection.cursor()
			cursor.execute(Queries.UNPROTECT % (local_path))
			connection.commit()

			connection.close()
		except DBError as e:
			print(f"DB error occured: {e}")


class Queries():
	CREATE_TABLE = """
		CREATE TABLE IF NOT EXISTS files (
  			file_id    TEXT PRIMARY KEY,
  			local_path TEXT NOT NULL,
  			using_now  INTEGER NOT NULL
		);
	"""
	GET_UNPROTECTED    = "SELECT local_path FROM files WHERE using_now=0;"
	DELETE_UNPROTECTED = "DELETE FROM files WHERE using_now=0;"
	PROTECT            = "UPDATE files SET using_now=using_now+1 WHERE local_path=\"%s\";"
	UNPROTECT          = "UPDATE files SET using_now=using_now-1 WHERE local_path=\"%s\";"
	GET_FILE_PATH      = "SELECT local_path FROM files WHERE file_id=\"%s\";"
	CREATE_ENTRY       = "INSERT INTO files (file_id, local_path, using_now) VALUES (\"%s\", \"%s\", 1);"
