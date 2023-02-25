from flask import Flask, request
import requests
import chat
import re
import datetime
import logging

from user import User

app = Flask(__name__)


formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = logging.FileHandler(f"logs/eloise-{datetime.date.today()}.txt")
handler.setFormatter(formatter)

logger = logging.getLogger("eloise")
logger.addHandler(handler)
logger.setLevel(logging.INFO)


banned_users = []
banned_groups = []
non_chat_groups = [143626394]

HELP_MESSAGE = '''Note: <text> means "any text"
It's recommanded to ASK her more!

---- FREE GENERATION ----
-h, -help: Show this help
-g, -gen <text>: Generate text
-t, -retry: Retry last generation
-m, -more: Continue generating more
'''

MORE_HELP_MESSAGE = '''
-qa <text>: Ask questions

---- CHAT WITH CONTEXT ----
-s, -reset: Reset your chat chain
-alt: Alternative reply
'''

CHAT_HELP_MESSAGE = '''
-c, -chat <text>: Chat with me
'''
PRIVATE_HELP_MESSAGE = '''
<text>: Chat with me
'''

received_messages = set()


def commands(user: User, message, enable_chat=False, is_private=False):
    help_match = re.match("\-h(elp)?", message)

    retry_match = re.match("\-(retry|t)", message)
    more_match = re.match("\-m(ore)?", message)
    gen_match = re.match("\-g(en)?\s+", message)
    qa_match = re.match("\-qa\s+", message)

    reset_match = re.match("\-(reset|s)", message)
    alt_match = re.match("\-alt", message)
    chat_match = re.match("\-c(hat)?\s+", message)

    help = HELP_MESSAGE
    if enable_chat:
        help += MORE_HELP_MESSAGE
    if enable_chat and not is_private:
        help += CHAT_HELP_MESSAGE
    if is_private:
        help += PRIVATE_HELP_MESSAGE

    prompt = message
    reply = ""
    matched = True

    if help_match:
        reply = help
    elif retry_match:
        reply = chat.on_generate(user, "", mode="retry")
    elif more_match:
        reply = chat.on_generate(user, "", mode="more")
    elif gen_match:
        prompt = message[gen_match.end():]
        reply = chat.on_generate(user, prompt, prompt)
    elif enable_chat and qa_match:
        prompt = message[qa_match.end():]
        reply = chat.on_generate(user, prompt, mode="qa")
    elif enable_chat and reset_match:
        reply = chat.on_reset(user)
    elif enable_chat and alt_match:
        reply = chat.on_message(user, "", alt=True)
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
            requests.get(
                f"http://127.0.0.1:5700/send_private_msg?user_id={user.id}&message={reply}")
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
            requests.get(
                f"http://127.0.0.1:5700/send_group_msg?group_id={group_id}&message=[CQ:at,qq={user.id}]\n{reply}")

    return 'OK'


if __name__ == '__main__':
    print("Starting server...")
    chat.init_run()
    app.run(debug=False, host='127.0.0.1', port=8000, threaded=False)
