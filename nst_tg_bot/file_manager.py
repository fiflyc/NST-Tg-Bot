import os
from aiogram import Bot
from aiogram.types.file import File
import sqlite3
from sqlite3 import Error as DBError
from threading import Timer
from nst_tg_bot.rwlock import RWLock


class FileManager():

	def __init__(self, bot: Bot,
		         cache_dir: str,
		         clearing_time:int,
		         path_to_db: str):
		self.__bot = bot
		self.__path_to_db = path_to_db + '/files.sqlite'
		self.__cache_dir = cache_dir
		self.__rw_lock = RWLock()

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
		self.__timer = Timer(self.__clearing_time, self.__clear_cache)
		self.__timer.start()

	def __clear_cache(self):
		self.__rw_lock.writer_acquire()
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

		self.__timer = Timer(self.__clearing_time, self.__clear_cache)
		self.__timer.start()

		self.__rw_lock.writer_release()

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
		self.__rw_lock.reader_acquire()
		local_path = None
		# unique single file code:
		file_id = file.file_unique_id

		try:
			connection = sqlite3.connect(self.__path_to_db)

			if not self.__is_loaded(file_id, connection):
				local_path = await self.__download_file(file)
				self.__create_entry(file_id, local_path, connection)
			else:
				local_path = self.__find_in_cache(file_id, connection)
				self.__protect_file(local_path, connection)

			connection.close()
		except DBError as e:
			print(f"DB error occured: {e}")

		self.__rw_lock.reader_release()

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
		self.__rw_lock.reader_acquire()

		try:
			connection = sqlite3.connect(self.__path_to_db)

			cursor = connection.cursor()
			cursor.execute(Queries.UNPROTECT % (local_path))
			connection.commit()

			connection.close()
		except DBError as e:
			print(f"DB error occured: {e}")

		self.__rw_lock.reader_release()


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
