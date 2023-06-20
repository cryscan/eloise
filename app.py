from flask import Flask, request
import os
import re
import requests
import datetime
import logging
from urllib.parse import quote
import markdown
import imgkit
import server
from server import User

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

received_messages = set()

IMAGE_THRESHOLD = 1024
IMAGE_WIDTH = 600


def sub_bullets(text):
    return re.sub('\n\* ([^\n]+)', '\n- \\1', text)


@app.route('/', methods=["POST"])
def handle_post():
    try:
        remote_addr = request.remote_addr
        json = request.get_json()
        type = json['message_type']
        message = json['raw_message']
        message_id = json['message_id']
        sender = json['sender']
        user = User(sender['user_id'], sender['nickname'], sender['sex'])
    except:
        return 'OK'

    if user in banned_users:
        return 'OK'
    if message_id in received_messages:
        return 'OK'
    if len(received_messages) > 500:
        received_messages.clear()

    if type == 'private':
        matched, prompt, reply = server.commands(
            user, message, enable_chat=True, is_private=True)

        if matched:
            logger.info(f"{user.nickname}({user.id}): {prompt}")
            logger.info(reply)
            received_messages.add(message_id)
            if len(reply) > IMAGE_THRESHOLD or reply.count('\n') > 2:
                options = {'font-family': 'SimSun'}
                html = markdown.markdown(
                    sub_bullets(reply), extensions=['extra', 'nl2br', 'sane_lists', 'smarty', 'codehilite'], options=options)

                file = f"./images/{user.id} {datetime.datetime.now().isoformat()}.png"
                file = file.replace(' ', '-')
                path = os.path.abspath(file)
                options = {'width': IMAGE_WIDTH}
                imgkit.from_string(
                    html, file, css='styles.css', options=options)
                requests.get(
                    f"http://{remote_addr}:5700/send_private_msg?user_id={user.id}&message=[CQ:image,file=file:///{path}]")
            else:
                requests.get(
                    f"http://{remote_addr}:5700/send_private_msg?user_id={user.id}&message={quote(reply)}")
    elif type == 'group':
        try:
            group_id = int(json['group_id'])
        except:
            return 'OK'
        if group_id in banned_groups:
            return 'OK'
        enable_chat = group_id not in non_chat_groups

        matched, prompt, reply = server.commands(
            user, message, enable_chat, is_private=False)
        if matched:
            logger.info(f"{group_id}: {user.nickname}({user.id}): {prompt}")
            logger.info(reply)
            received_messages.add(message_id)
            if len(reply) > IMAGE_THRESHOLD or reply.count('\n') > 2:
                options = {'font-family': 'SimSun'}
                html = markdown.markdown(
                    sub_bullets(reply), extensions=['extra', 'nl2br', 'sane_lists', 'smarty', 'codehilite'], options=options)

                file = f"./images/{user.id} {datetime.datetime.now().isoformat()}.png"
                file = file.replace(' ', '-')
                path = os.path.abspath(file)
                options = {'width': IMAGE_WIDTH}
                imgkit.from_string(
                    html, file, css='styles.css', options=options)
                requests.get(
                    f"http://{remote_addr}:5700/send_group_msg?group_id={group_id}&message=[CQ:reply,id={message_id}][CQ:image,file=file:///{path}]")
            else:
                requests.get(
                    f"http://{remote_addr}:5700/send_group_msg?group_id={group_id}&message=[CQ:reply,id={message_id}]{quote(reply)}")

    return 'OK'


@app.route('/chat', methods=['GET'])
def handle_chat():
    try:
        args = request.args
        user_id = args['user_id']
        user_nickname = args.get('user_nickname', 'John')
        user_sex = args.get('user_sex', 'unknown')
        rendered = args.get('rendered')

        temp = args.get('temp')
        top_p = args.get('top_p')
        tau = args.get('tau')
        af = args.get('af')
        ap = args.get('ap')

        message = args['message']
    except:
        return ''

    if temp:
        message += f'-temp={temp} '
    if top_p:
        message += f'-top_p={top_p} '
    if tau:
        message += f'-tau={tau} '
    if af:
        message += f'-af={af} '
    if ap:
        message += f'-ap={ap} '

    user = server.User(user_id, user_nickname, user_sex)
    matched, prompt, reply = server.commands(
        user, message, enable_chat=True, is_private=True)

    if matched:
        logger.info(f"{user.nickname}({user.id}): {prompt}")
        logger.info(reply)
        if rendered:
            options = {'font-family': 'SimSun'}
            html = markdown.markdown(
                sub_bullets(reply), extensions=['extra', 'nl2br', 'sane_lists', 'smarty', 'codehilite'], options=options)
            return html
        else:
            return reply

    return ''


if __name__ == '__main__':
    print("Starting server...")
    server.init()
    app.run(debug=False, host='127.0.0.1', port=6006, threaded=False)
