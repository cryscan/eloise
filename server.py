from flask import Flask, request
import requests

import chat
import re

app = Flask(__name__)


banned_groups = []
chat_groups = [1058164126]

HELP_MESSAGE = '''Commands:

+h(elp): Show this help

+g(en) <text>: Generate text

+retry|t: Retry last generation

+m(ore): Continue generating more
'''

MORE_HELP_MESSAGE = '''

+qa: Ask questions

+c(hat) <text>: Chat with me

+reset|s: Reset your chat chain

+alt: Alternative reply
'''


def gen_commands(uid, message, enable_qa=False):
    gen_match = re.match("\+g(en)?\s+", message)
    qa_match = re.match("\+qa\s+", message)

    matched = False
    reply = ""
    if gen_match is not None:
        reply = chat.on_generate(uid, message[gen_match.end():])
        matched = True
    elif enable_qa and qa_match is not None:
        reply = chat.on_generate(uid, message[qa_match.end():], mode="qa")
        matched = True
    elif re.match("\+(retry|t)", message) is not None:
        reply = chat.on_generate(uid, "", mode="retry")
        matched = True
    elif re.match("\+m(ore)?", message) is not None:
        reply = chat.on_generate(uid, "", mode="more")
        matched = True
    return reply, matched


@app.route('/', methods=["POST", "GET"])
def post_data():
    json = request.get_json()
    type = json.get('message_type')
    message = json.get('raw_message')
    sender = json.get('sender')

    user_id = None
    group_id = None
    nickname = None

    reply = ""

    if type == 'private':
        user_id = sender.get('user_id')
        nickname = sender.get('nickname')

        if re.match("\+h(elp)?", message) is not None:
            reply = HELP_MESSAGE + MORE_HELP_MESSAGE
        elif re.match("\+(reset|s)", message) is not None:
            reply = chat.on_reset(user_id, nickname)
        elif re.match("\+alt", message) is not None:
            reply = chat.on_message(user_id, "", True)
        else:
            reply, matched = gen_commands(user_id, message, enable_qa=True)
            if not matched:
                print(f"{nickname}({user_id}): {message}")
                reply = chat.on_message(user_id, message, False)

        print(reply)
        requests.get(
            f"http://127.0.0.1:5700/send_private_msg?user_id={user_id}&message={reply}")
    elif type == 'group':
        group_id = int(json.get('group_id'))
        user_id = sender.get('user_id')
        nickname = sender.get('nickname')

        help = HELP_MESSAGE

        if group_id in banned_groups:
            return 'OK'

        full_function = group_id in chat_groups
        reply, matched = gen_commands(
            user_id, message, enable_qa=full_function)
        if not matched:
            if full_function:
                # Full chat functionalities
                help += MORE_HELP_MESSAGE

                if re.match("\+(reset|s)", message) is not None:
                    reply = chat.on_reset(user_id, nickname)
                elif re.match("\+alt", message) is not None:
                    reply = chat.on_message(user_id, "", alt=True)
                else:
                    print(f"{nickname}({user_id}): {message}")
                    chat_match = re.match("\+c(hat)?\s+", message)
                    if chat_match is not None:
                        reply = chat.on_message(
                            user_id, message[chat_match.end():])

            if re.match("\+h(elp)?", message) is not None:
                reply = help

        print(reply)
        requests.get(
            f"http://127.0.0.1:5700/send_group_msg?group_id={group_id}&message={reply}")

    return 'OK'


if __name__ == '__main__':
    print("Start server...")
    chat.load_model()
    chat.init_run()

    app.run(debug=False, host='127.0.0.1', port=8000, threaded=False)
