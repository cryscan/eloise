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

from prompt import User, SCENARIO_ALICE, SCENARIO_ELOISE
from chat import GenerateMode, CHAT_SAMPLER, INSTRUCT_SAMPLER, model

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


try:
    with open("qq.txt", 'r') as file:
        QQ = file.read()
except:
    print("Please provide your QQ number in `qq.txt`")
    QQ = ""

IMAGE_THRESHOLD = 1024
IMAGE_WIDTH = 600

CHAT_HELP_COMMAND = "`-c, -chat <text>`"
PRIVATE_HELP_COMMAND = "`<text>`"

with open("./help.md", 'r') as file:
    model_name = model.args.MODEL_NAME.split('/')[-1].replace('.pth', '')

    HELP_MESSAGE = file.read()
    HELP_MESSAGE = HELP_MESSAGE.replace('<model>', model_name)
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_nucleus>', 'Yes' if CHAT_SAMPLER.sample.__name__ == "sample_nucleus" else '')
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_typical>', 'Yes' if CHAT_SAMPLER.sample.__name__ == "sample_typical" else '')
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_temp>', str(CHAT_SAMPLER.temp))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_top_p>', str(CHAT_SAMPLER.top_p))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_tau>', str(CHAT_SAMPLER.tau))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_af>', str(CHAT_SAMPLER.count_penalty))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_ap>', str(CHAT_SAMPLER.presence_penalty))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<chat_ar>', str(CHAT_SAMPLER.penalty_range))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_nucleus>', 'Yes' if INSTRUCT_SAMPLER.sample.__name__ == "sample_nucleus" else '')
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_typical>', 'Yes' if INSTRUCT_SAMPLER.sample.__name__ == "sample_typical" else '')
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_temp>', str(INSTRUCT_SAMPLER.temp))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_top_p>', str(INSTRUCT_SAMPLER.top_p))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_tau>', str(INSTRUCT_SAMPLER.tau))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_af>', str(INSTRUCT_SAMPLER.count_penalty))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_ap>', str(INSTRUCT_SAMPLER.presence_penalty))
    HELP_MESSAGE = HELP_MESSAGE.replace(
        '<inst_ar>', str(INSTRUCT_SAMPLER.penalty_range))

received_messages = set()


def commands(user: User, message, enable_chat=False, is_private=False):
    help_match = re.match("\-h(elp)?", message)
    params_match = re.match("\-p(arams)?", message)

    translate_match = re.match("\-tr", message)
    retry_match = re.match("\-(retry|e)", message)
    more_match = re.match("\-m(ore)?", message)
    gen_match = re.match("\-g(en)?\s+", message)
    inst_match = re.match("\-i(nst)?\s+", message)

    reset_match = re.match("\-(reset|s)", message)
    reset_bot_match = re.match("\-(bot|b)", message)
    alt_match = re.match("\-a(lt)?", message)
    chat_match = re.match("\-c(hat)?\s+", message)
    at_match = re.match(f"\[CQ:at,qq={QQ}\]", message)

    help = HELP_MESSAGE
    if is_private:
        help = help.replace('<chat>', PRIVATE_HELP_COMMAND)
    else:
        help = help.replace('<chat>', CHAT_HELP_COMMAND)

    prompt = message
    reply = ""
    matched = True

    if help_match:
        reply = help
    elif params_match:
        prompt = message[params_match.end():]
        reply = chat.on_show_params(user, prompt)
    elif translate_match:
        prompt = message[translate_match.end():]
        reply = chat.on_translate(user, prompt)
    elif retry_match:
        prompt = message[retry_match.end():]
        reply = chat.on_generate(user, prompt, mode=GenerateMode.RETRY)
    elif more_match:
        prompt = message[more_match.end():]
        reply = chat.on_generate(user, prompt, mode=GenerateMode.MORE)
    elif gen_match:
        prompt = message[gen_match.end():]
        reply = chat.on_generate(user, prompt, mode=GenerateMode.GENERATE)
    elif enable_chat and inst_match:
        prompt = message[inst_match.end():]
        reply = chat.on_generate(user, prompt, mode=GenerateMode.INSTRUCT)
    elif enable_chat and reset_bot_match:
        prompt = message[reset_bot_match.end():]
        reply = chat.on_reset(user, prompt, SCENARIO_ALICE, INSTRUCT_SAMPLER)
    elif enable_chat and reset_match:
        prompt = message[reset_match.end():]
        reply = chat.on_reset(user, prompt, SCENARIO_ELOISE, CHAT_SAMPLER)
    elif enable_chat and alt_match:
        prompt = message[alt_match.end():]
        reply = chat.on_message(user, prompt, alt=True)
    elif enable_chat and is_private:
        reply = chat.on_message(user, prompt)
    elif enable_chat and not is_private and chat_match:
        prompt = message[chat_match.end():]
        reply = chat.on_message(user, prompt)
    elif QQ and enable_chat and not is_private and at_match:
        prompt = message[at_match.end():]
        reply = chat.on_message(user, prompt)
    else:
        matched = False

    return matched, prompt, reply


@app.route('/', methods=["POST"])
def handle_post():
    try:
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
        matched, prompt, reply = commands(
            user, message, enable_chat=True, is_private=True)

        if matched:
            logger.info(f"{user.nickname}({user.id}): {prompt}")
            logger.info(reply)
            received_messages.add(message_id)
            if len(reply) > IMAGE_THRESHOLD or reply.count('\n') > 2:
                options = {'font-family': 'SimSun'}
                html = markdown.markdown(
                    reply, extensions=['extra', 'nl2br', 'sane_lists', 'codehilite'], options=options)

                file = f"./images/{user.id} {datetime.datetime.now().isoformat()}.png"
                file = file.replace(' ', '-')
                path = os.path.abspath(file)
                options = {'width': IMAGE_WIDTH}
                imgkit.from_string(
                    html, file, css='styles.css', options=options)
                requests.get(
                    f"http://127.0.0.1:5700/send_private_msg?user_id={user.id}&message=[CQ:image,file=file:///{path}]")
            else:
                requests.get(
                    f"http://127.0.0.1:5700/send_private_msg?user_id={user.id}&message={quote(reply)}")
    elif type == 'group':
        try:
            group_id = int(json['group_id'])
        except:
            return 'OK'
        if group_id in banned_groups:
            return 'OK'
        enable_chat = group_id not in non_chat_groups

        matched, prompt, reply = commands(
            user, message, enable_chat, is_private=False)
        if matched:
            logger.info(f"{group_id}: {user.nickname}({user.id}): {prompt}")
            logger.info(reply)
            received_messages.add(message_id)
            if len(reply) > IMAGE_THRESHOLD or reply.count('\n') > 2:
                options = {'font-family': 'SimSun'}
                html = markdown.markdown(
                    reply, extensions=['extra', 'nl2br', 'sane_lists', 'codehilite'], options=options)

                file = f"./images/{user.id} {datetime.datetime.now().isoformat()}.png"
                file = file.replace(' ', '-')
                path = os.path.abspath(file)
                options = {'width': IMAGE_WIDTH}
                imgkit.from_string(
                    html, file, css='styles.css', options=options)
                requests.get(
                    f"http://127.0.0.1:5700/send_group_msg?group_id={group_id}&message=[CQ:reply,id={message_id}][CQ:image,file=file:///{path}]")
            else:
                requests.get(
                    f"http://127.0.0.1:5700/send_group_msg?group_id={group_id}&message=[CQ:reply,id={message_id}]{quote(reply)}")

    return 'OK'


if __name__ == '__main__':
    print("Starting server...")
    chat.init_run()
    app.run(debug=False, host='127.0.0.1', port=8000, threaded=False)
