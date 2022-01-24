from threading import Lock, Condition


class RWLock():

	def __init__(self):
		self.__readers = 0
		self.__lock_read = Lock()
		self.__lock_write = Lock()
		self.__all_readers_waiting = Condition(self.__lock_read)

	def reader_acquire(self):
		self.__lock_write.acquire()
		self.__lock_read.acquire()

		self.__readers += 1

		self.__lock_read.release()
		self.__lock_write.release()

	def reader_release(self):
		self.__lock_read.acquire()
		
		self.__readers -= 1
		if self.__readers == 0:
			self.__all_readers_waiting.notify_all()

		self.__lock_read.release()

	def writer_acquire(self):
		self.__lock_write.acquire()

		with self.__all_readers_waiting:
			self.__all_readers_waiting.wait_for(lambda: self.__readers == 0)

	def writer_release(self):
		self.__lock_write.release()
