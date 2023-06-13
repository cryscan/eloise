import chat
import re

from prompt import User, SCENARIOS, CHAT_SAMPLER, INSTRUCT_SAMPLER
from chat import GenerateMode, model


try:
    with open("qq.txt", 'r') as file:
        QQ = file.read()
except:
    print("Please provide your QQ number in `qq.txt`")
    QQ = ""

CHAT_HELP_COMMAND = "-c, -chat"
PRIVATE_HELP_COMMAND = ""

with open("./help.md", 'r') as file:
    model_name = model.args.MODEL_NAME.split('/')[-1].replace('.pth', '')

    HELP_MESSAGE = file.read()
    HELP_MESSAGE = HELP_MESSAGE.replace('<model>', model_name)
    HELP_MESSAGE = HELP_MESSAGE.replace('<scenarios>', str(SCENARIOS))
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


def commands(user: User, message, enable_chat=False, is_private=False):
    help_match = re.match("\-h(elp)?", message)
    params_match = re.match("\-p(arams)?", message)
    prompts_match = re.match("\-pr(ompts)?", message)

    translate_match = re.match("\-tr", message)
    retry_match = re.match("\-(retry|e)", message)
    more_match = re.match("\-m(ore)?", message)
    gen_match = re.match("\-g(en)?\s+", message)
    inst_match = re.match("\-i(nst)?\s+", message)

    reset_match = re.match("\-(reset|s)\s*", message)
    list_match = re.match("\-l(ist)?", message)
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
    elif prompts_match:
        prompt = message[prompts_match.end():]
        reply = chat.on_show_params(user, prompt, prompts=True)
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
    elif enable_chat and reset_match:
        prompt = message[reset_match.end():]
        reply = chat.on_reset(user, prompt)
    elif enable_chat and list_match:
        reply = str(SCENARIOS)
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


def init():
    chat.init_run()
