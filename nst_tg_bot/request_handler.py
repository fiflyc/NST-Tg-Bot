from enum import Enum
import os
from aiogram.types.file import File
import sqlite3
from sqlite3 import Error as DBError
from threading import Timer
from nst_tg_bot.file_manager import FileManager


class InputType(Enum):
	CONTENT = 0,
	STYLE = 1


class RequestHandler():

	def __init__(self,
		         file_manager: FileManager,
		         path_to_db: str,
		         wait_time: int):
		self.__file_manager = file_manager
		self.__path_to_db = path_to_db + "/queries.sqlite"

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

		# self.__wait_time = wait_time
		# self.__timer = Timer(self.__clearing_time, self.__clear_cache)
		# self.__timer.start()

	async def set_input(self, in_type: InputType, file: File, chat_id: int):
		file_path = await self.__file_manager.get_local_path(file)

		try:
			connection = sqlite3.connect(self.__path_to_db)

			if self.__is_new_query(chat_id, connection):
				self.__create_entry(chat_id, connection)
			if in_type is InputType.CONTENT:
				self.__set_content(chat_id, file_path, connection)
			if in_type is InputType.STYLE:
				self.__set_style(chat_id, file_path, connection)

			connection.close()
		except DBError as e:
			print(f"DB error occured: {e}")

	def __is_new_query(self, chat_id: int, connection):
		cursor = connection.cursor()
		cursor.execute(Queries.GET_INPUT % chat_id)

		return len(cursor.fetchall()) == 0

	def __create_entry(self, chat_id: int, connection):
		cursor = connection.cursor()
		cursor.execute(Queries.CREATE_ENTRY % chat_id)
		connection.commit()

	def __set_content(self, chat_id: int, file_path: str, connection):
		cursor = connection.cursor()
		cursor.execute(Queries.SET_CONTENT % (file_path, chat_id))
		connection.commit()

	def __set_style(self, chat_id: int, file_path: str, connection):
		cursor = connection.cursor()
		cursor.execute(Queries.SET_STYLE % (file_path, chat_id))
		connection.commit()

	def execute_query(self, chat_id: int):
		pass


class Queries():
	CREATE_TABLE = """
		CREATE TABLE IF NOT EXISTS queries (
  			chat_id INT PRIMARY KEY,
  			content TEXT,
  			style   TEXT,
  			marked  INTEGER NOT NULL
		);
	"""
	MARK          = "UPDATE queries SET marked=1 WHERE marked=0;"
	DELETE_MARKED = "DELETE FROM queries WHERE marked=1;"
	GET_MARKED    = "SELECT content, style FROM queries WHERE marked=1;"
	DELETE        = "DELETE FROM queries WHERE chat_id=%d;"
	GET_INPUT     = "SELECT content, style FROM queries WHERE chat_id=%d;"
	CREATE_ENTRY  = "INSERT INTO queries (chat_id, marked) VALUES (%d, 0);"
	SET_CONTENT   = "UPDATE queries SET content=\"%s\" WHERE chat_id=%d;"
	SET_STYLE     = "UPDATE queries SET style=\"%s\" WHERE chat_id=%d;"
