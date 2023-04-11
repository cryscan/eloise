from flask import Flask, request
import os
import requests
import chat
import re
import datetime
import logging
from urllib.parse import quote
import markdown
import imgkit

from prompt import User

app = Flask(__name__)


formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = logging.FileHandler(f"logs/eloise-{datetime.date.today()}.txt")
handler.setFormatter(formatter)

logger = logging.getLogger("eloise")
logger.addHandler(handler)
logger.setLevel(logging.INFO)


banned_users = []
banned_groups = []
non_chat_groups = []


IMAGE_THRESHOLD = 300
IMAGE_WIDTH = 400


with open("./help.md", 'r') as file:
    HELP_MESSAGE = file.read()
CHAT_HELP_COMMAND = "`-c, -chat <text>`"
PRIVATE_HELP_COMMAND = "`<text>`"
MODEL_NAME = "Raven v8 14B EngAndMore ctx 4096"

received_messages = set()


def commands(user: User, message, enable_chat=False, is_private=False):
    help_match = re.match("\-h(elp)?", message)

    translate_match = re.match("\-tr", message)
    retry_match = re.match("\-(retry|e)", message)
    more_match = re.match("\-m(ore)?", message)
    gen_match = re.match("\-g(en)?\s+", message)
    qa_match = re.match("\-qa\s+", message)
    inst_match = re.match("\-i(nst)?\s+", message)

    reset_match = re.match("\-(reset|s)", message)
    reset_bot_match = re.match("\-(bot|b)", message)
    alt_match = re.match("\-alt", message)
    chat_match = re.match("\-c(hat)?\s+", message)

    help = HELP_MESSAGE.replace('<model>', MODEL_NAME)
    if is_private:
        help = help.replace('<chat>', PRIVATE_HELP_COMMAND)
    else:
        help = help.replace('<chat>', CHAT_HELP_COMMAND)

    prompt = message
    reply = ""
    matched = True

    if help_match:
        reply = help
    elif translate_match:
        prompt = message[translate_match.end():]
        reply = chat.on_translate(user, prompt)
    elif retry_match:
        reply = chat.on_generate(user, prompt, mode="retry")
    elif more_match:
        reply = chat.on_generate(user, prompt, mode="more")
    elif gen_match:
        prompt = message[gen_match.end():]
        reply = chat.on_generate(user, prompt, prompt)
    elif enable_chat and qa_match:
        prompt = message[qa_match.end():]
        reply = chat.on_generate(user, prompt, mode="qa")
    elif enable_chat and inst_match:
        prompt = message[inst_match.end():]
        reply = chat.on_generate(user, prompt, mode="inst")
    elif enable_chat and reset_bot_match:
        reply = chat.on_reset_bot(user)
    elif enable_chat and reset_match:
        reply = chat.on_reset(user)
    elif enable_chat and alt_match:
        reply = chat.on_message(user, prompt, alt=True)
    elif enable_chat and is_private:
        reply = chat.on_message(user, prompt)
    elif enable_chat and not is_private and chat_match:
        prompt = message[chat_match.end():]
        reply = chat.on_message(user, prompt)
    else:
        matched = False

    return matched, prompt, reply


@app.route('/', methods=["POST"])
def post_data():
    json = request.get_json()
    type = json.get('message_type')
    message = json.get('raw_message')
    message_id = json.get('message_id')
    sender = json.get('sender')

    if message_id in received_messages:
        return 'OK'

    if type == 'private':
        user = User(sender)
        if user in banned_users:
            return 'OK'

        matched, prompt, reply = commands(
            user, message, enable_chat=True, is_private=True)

        if matched:
            logger.info(f"{user.nickname}({user.id}): {prompt}")
            logger.info(reply)
            received_messages.add(message_id)
            if len(reply) > IMAGE_THRESHOLD or reply.count('\n') > 2:
                options = {
                    'width': IMAGE_WIDTH, 
                    'disable-smart-width': '',
                    'font-family': 'SimSun',
                }
                html = markdown.markdown(
                    reply, extensions=['extra', 'nl2br'], options=options)

                file = f"./images/{user.id} {datetime.datetime.now().isoformat()}.png"
                file = file.replace(' ', '-')

                path = os.path.abspath(file)
                imgkit.from_string(html, file)
                requests.get(
                    f"http://127.0.0.1:5700/send_private_msg?user_id={user.id}&message=[CQ:image,file=file:///{path}]")
            else:
                requests.get(
                    f"http://127.0.0.1:5700/send_private_msg?user_id={user.id}&message={quote(reply)}")
    elif type == 'group':
        group_id = int(json.get('group_id'))
        if group_id in banned_groups:
            return 'OK'
        enable_chat = group_id not in non_chat_groups

        user = User(sender)
        if user.id in banned_users:
            return 'OK'

        matched, prompt, reply = commands(
            user, message, enable_chat, is_private=False)
        if matched:
            logger.info(f"{group_id}: {user.nickname}({user.id}): {prompt}")
            logger.info(reply)
            received_messages.add(message_id)
            if len(reply) > IMAGE_THRESHOLD or reply.count('\n') > 2:
                options = {
                    'width': IMAGE_WIDTH, 
                    'disable-smart-width': '',
                    'font-family': 'SimSun',
                }
                html = markdown.markdown(
                    reply, extensions=['extra', 'nl2br'], options=options)

                file = f"./images/{user.id} {datetime.datetime.now().isoformat()}.png"
                file = file.replace(' ', '-')

                path = os.path.abspath(file)
                imgkit.from_string(html, file)
                requests.get(
                    f"http://127.0.0.1:5700/send_group_msg?group_id={group_id}&message=[CQ:at,qq={user.id}][CQ:image,file=file:///{path}]")
            else:
                requests.get(
                    f"http://127.0.0.1:5700/send_group_msg?group_id={group_id}&message=[CQ:at,qq={user.id}]\n{quote(reply)}")

    return 'OK'


if __name__ == '__main__':
    print("Starting server...")
    chat.init_run()
    app.run(debug=False, host='127.0.0.1', port=8000, threaded=False)
