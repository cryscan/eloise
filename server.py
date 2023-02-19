from flask import Flask, request
import requests

import chat
import re

app = Flask(__name__)


@app.route('/', methods=["POST", "GET"])
def post_data():
    json = request.get_json()
    type = json.get('message_type')
    message = json.get('raw_message')
    sender = json.get('sender')

    user_id = None
    group_id = None

    if type == 'private':
        user_id = sender.get('user_id')
        print(f"{user_id}: {message}")

        if re.match("\+reset", message) is not None:
            reply = chat.on_reset(user_id)
        elif re.match("\+retry", message) is not None:
            reply = chat.on_message(user_id, "", True)
        else:
            reply = chat.on_message(user_id, message, False)
        print(reply)

        requests.get(
            f"http://127.0.0.1:5700/send_private_msg?user_id={user_id}&message={reply}")
    elif type == 'group':
        group_id = json.get('group_id')
        user_id = sender.get('user_id')

        if re.match("\+reset", message) is not None:
            reply = chat.on_reset(user_id)
        elif re.match("\+retry", message) is not None:
            reply = chat.on_message(user_id, "", True)
        else:
            _, span = re.match("\+chat\s", message)
            reply = chat.on_message(user_id, message[span:], False)
        print(reply)

        requests.get(
            f"http://127.0.0.1:5700/send_group_msg?group_id={group_id}&message={reply}")
    return 'OK'


if __name__ == '__main__':
    print("Start server...")
    chat.load_model()
    chat.init_run()

    app.run(debug=False, host='127.0.0.1', port=8000)
