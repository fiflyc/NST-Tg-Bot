import argparse
import os
import nst_tg_bot.config as cfg


def extract_args():
	parser = argparse.ArgumentParser("FiFlyC's NST Telegram Bot")
	parser.add_argument('--api_token', '-t', type=str,
	                    help="API token from @BotFather.")
	parser.add_argument('--downloads_path', '-d', type=str, default="./saved",
	                    help="Path for Telegram's downloads.")
	parser.add_argument('--db_path', '-b', type=str, default="./db",
	                    help="Path for sqlite databases.")
	parser.add_argument('--clearing_time', '-c', type=int, default=7200,
	                    help="Clearing cache time in seconds.")
	parser.add_argument('--waiting_time', '-w', type=int, default=1200,
	                    help="Waiting for user's second image. " + \
	                         "If image wasn't received before that time, " + \
	                         "bot may forget user's previous query.")
	args = parser.parse_args()

	if args.api_token is None:
		raise ValueError("API token is necessary for launching the bot. Use --api_token [TOKEN]")
	if args.clearing_time < 0:
		raise ValueError("Clearing cache time should be greater than 0.")
	if args.waiting_time < 0:
		raise ValueError("Waiting user's responce time should be greater than 0.")

	return args


def update_config(args):
    if not os.path.exists(args.downloads_path):
    	os.mkdir(args.downloads_path)
    if not os.path.exists(args.db_path):
    	os.mkdir(args.db_path)
    
    cfg.TOKEN         = args.api_token
    cfg.DOWNLOADS     = args.downloads_path
    cfg.DATABASES     = args.db_path
    cfg.CLEARING_TIME = args.clearing_time
    cfg.WAITING_TIME  = args.waiting_time


if __name__ == '__main__':
	args = extract_args()
	update_config(args)
	exec(open('./nst_tg_bot/nst_bot.py').read())
