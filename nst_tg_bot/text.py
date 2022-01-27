START_MESSAGE = """
This bot can transfer image style from an artwork to a photo.
Use /help for more details.
"""

HELP_MESSAGE = """
*Usage*

Write /content in an image description to change it's style.
You can also type /content in a reply to message with an image.
Do the same thing with /style passing an artwotk which style you want to use.

*Commands*

/help - show this list
/info - show some information about bot
/content - send image to change
/style - send image with target style
/links - show links with more info
"""
LINKS_MESSAGE = """
*Links*

[Article with algorithm](https://arxiv.org/abs/1612.04337)
[Source code](https://github.com/fiflyc/NST-Tg-Bot)
"""

INFO_MESSAGE      = "This bot can transfer image style from an artwork to a photo. Use /help for more details."
NO_CONTENT_IMAGE  = "Use /content in a photo description or in a reply to message with the photo"
NO_STYLE_IMAGE    = "Use /style in a photo description or in a reply to message with the photo"
SEND_STYLE        = "Now send me an artwork with a style to transfer"
SEND_CONTENT      = "Now send me a photo to change it's style"
WRONG_FILE_FORMAT = "The file is not an image"
QUERY_RECEIVED    = "Now I will transfer style to your image. Approximate waiting time: 90 seconds"
