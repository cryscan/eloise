from flask import Flask, request
import requests
import chat
import re

from user import User

app = Flask(__name__)


banned_groups = []
chat_groups = [1058164126, 1079546706, 686922858]

HELP_MESSAGE = '''Commands:

-h(elp): Show this help

-g(en) <text>: Generate text

-retry|t: Retry last generation

-m(ore): Continue generating more
'''

MORE_HELP_MESSAGE = '''
-qa <text>: Ask questions

-reset|s: Reset your chat chain

-alt: Alternative reply
'''

CHAT_HELP_MESSAGE = '''
-c(hat) <text>: Chat with me
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

    reply = ""
    matched = True

    if help_match:
        reply = help
    elif retry_match:
        reply = chat.on_generate(user, "", mode="retry")
    elif more_match:
        reply = chat.on_generate(user, "", mode="more")
    elif gen_match:
        reply = chat.on_generate(user, message[gen_match.end():])
    elif enable_chat and qa_match:
        reply = chat.on_generate(user, message[qa_match.end():], mode="qa")
    elif enable_chat and reset_match:
        reply = chat.on_reset(user)
    elif enable_chat and alt_match:
        reply = chat.on_message(user, "", alt=True)
    elif enable_chat and is_private:
        reply = chat.on_message(user, message)
    elif enable_chat and not is_private and chat_match:
        reply = chat.on_message(user, message[chat_match.end():])
    else:
        matched = False

    return reply, matched


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
        reply, matched = commands(
            user, message, enable_chat=True, is_private=True)

        if matched:
            print(f"{user.nickname}({user.id}): {message}")
            print(reply)

            received_messages.add(message_id)
            requests.get(
                f"http://127.0.0.1:5700/send_private_msg?user_id={user.id}&message={reply}")
    elif type == 'group':
        group_id = int(json.get('group_id'))
        if group_id in banned_groups:
            return 'OK'
        enable_chat = group_id in chat_groups

        user = User(sender)
        reply, matched = commands(user, message, enable_chat, is_private=False)
        if matched:
            print(f"{group_id}: {user.nickname}({user.id}): {message}")
            print(reply)

            received_messages.add(message_id)
            requests.get(
                f"http://127.0.0.1:5700/send_group_msg?group_id={group_id}&message={reply}")

    return 'OK'


if __name__ == '__main__':
    print("Starting server...")
    chat.init_run()
    app.run(debug=False, host='127.0.0.1', port=8000, threaded=False)
