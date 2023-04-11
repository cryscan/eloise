import os
import copy
import sys
import time
import types
import gc
import re
import numpy as np
import torch
import pickle
import translate
import langid

from model.model_run import RWKV
from model.utils import TOKENIZER

import prompt
from prompt import User

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# '1' or '0', please use torch 1.13+ and benchmark speed
os.environ["RWKV_JIT_ON"] = '1'
# '1' : use CUDA kernel for seq mode (much faster)
os.environ["RWKV_CUDA_ON"] = '1'

CHAT_LANG = 'English'  # English Chinese
# CHAT_LANG = 'Chinese'
SAME_LANG = "PLEASE SELECT TWO DISTINCT LANGUAGES"

tokenizer = TOKENIZER("20B_tokenizer.json")

DONT_OUTPUT = -float('inf')

MAX_MESSAGE_LEN = 4096
CHUNK_LEN = 128

MAX_GENERATE_LEN = 250
MAX_REPLY_LEN = 1024

args = types.SimpleNamespace()

# args.strategy = 'cpu fp32'
# args.strategy = 'cuda fp16'
# args.strategy = 'cuda fp16 *8 -> cpu fp32'
# args.strategy = 'cuda fp16 *6+'
# args.strategy = 'cuda fp16 *0+ -> cpu fp32 *1'
# args.strategy = 'cuda fp16 *32 -> cpu fp32'
# args.strategy = 'cuda fp16 *20 -> cpu fp32'
args.strategy = 'cuda fp16i8 *20 -> cuda fp16'

# args.MODEL_NAME = '/root/autodl-tmp/models/RWKV-4-Pile-7B-20221115-8047'
# args.MODEL_NAME = '/root/autodl-tmp/models/RWKV-4-Pile-14B-20230213-8019'
# args.MODEL_NAME = '/root/autodl-tmp/models/RWKV-4-Pile-14B-20230228-ctx4096-test663'
# args.MODEL_NAME = '/root/autodl-tmp/models/RWKV-4-Pile-14B-20230313-ctx8192-test1050'
# args.MODEL_NAME = '/root/autodl-tmp/models/RWKV-4-Pile-14B-Instruct-test5-20230329-ctx4096'
args.MODEL_NAME = '/root/autodl-tmp/models/RWKV-4-Raven-14B-v8-EngAndMore-20230408-ctx4096'
# args.MODEL_NAME = '/root/autodl-tmp/models/RWKV-4-Pile-7B-EngChn-test5-20230330'

args.STATE_DUMP_NAME = './state_14b'

args.vocab_size = 50277
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0

args.n_layer = 40   # 32
args.n_embd = 5120  # 4096
args.ctx_len = 4096


# Load Model
print(f"Loading... {args.MODEL_NAME}")
# os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE
model = RWKV(model=args.MODEL_NAME, strategy=args.strategy)


def run_rnn(tokens, model_state=None):
    tokens = [int(x) for x in tokens]

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    return out, model_state


all_state = {}


def clean_user_state(uid, channel):
    n = f'{uid}_{channel}'
    if n in all_state.keys():
        del all_state[n]


def save_all_state(uid, channel, last_out, model_state, model_tokens):
    n = f'{uid}_{channel}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['state'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)


def load_all_state(uid, channel):
    n = f'{uid}_{channel}'
    model_state = copy.deepcopy(all_state[n]['state'])
    model_tokens = copy.deepcopy(all_state[n]['token'])

    if model_state:
        for i in range(args.n_layer):
            dd = model.strategy[i]
            dev = dd.device
            atype = dd.atype
            model_state[i*5+0] = model_state[i*5+0].to(atype).to(dev)
            model_state[i*5+1] = model_state[i*5+1].to(torch.float).to(dev)
            model_state[i*5+2] = model_state[i*5+2].to(torch.float).to(dev)
            model_state[i*5+3] = model_state[i*5+3].to(torch.float).to(dev)
            model_state[i*5+4] = model_state[i*5+4].to(atype).to(dev)

    return all_state[n]['out'], model_state, model_tokens


def save_arguments(uid, channel, **kwargs):
    n = f'arguments_{uid}_{channel}'
    all_state[n] = kwargs


def load_arguments(uid, channel):
    n = f'arguments_{uid}_{channel}'
    return all_state[n]


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def init_run():
    try:
        recover_all_state()
        print("Recovered state")
    except:
        print("Loading chat intro...")
        tokens = tokenizer.encode(prompt.default_user.chat_intro())
        out, state = run_rnn(tokens)
        save_all_state("", "chat_intro", out, state, tokens)

        print("Loading bot chat intro...")
        tokens = tokenizer.encode(prompt.default_user.chat_intro_bot())
        out, state = run_rnn(tokens)
        save_all_state("", "chat_intro_bot", out, state, tokens)

        clear_cache()
        dump_all_state()


def recover_all_state():
    global all_state
    with open(args.STATE_DUMP_NAME + '.pickle', 'rb') as file:
        all_state = pickle.load(file)


def dump_all_state():
    with open(args.STATE_DUMP_NAME + '.pickle', 'wb') as file:
        pickle.dump(all_state, file, protocol=pickle.HIGHEST_PROTOCOL)


def clamp(n, minimum, maximum):
    return max(minimum, min(n, maximum))


def read_sampler_params(message: str, temp, top_p, count_penalty, presence_penalty):
    temp_match = re.search("(\-temp\s*=\s*)([^\s]+)\s+", message)
    top_p_match = re.search("(\-top_p\s*=\s*)([^\s]+)\s+", message)
    af_match = re.search("(\-af\s*=\s*)([^\s]+)\s+", message)
    ap_match = re.search("(\-ap\s*=\s*)([^\s]+)\s+", message)

    if temp_match:
        temp = float(temp_match.group(2))
        message = message.replace(temp_match.group(0), "")
        print(f"temp: {temp}")
    if top_p_match:
        top_p = float(top_p_match.group(2))
        message = message.replace(top_p_match.group(0), "")
        print(f"top_p: {top_p}")
    if af_match:
        count_penalty = float(af_match.group(2))
        message = message.replace(af_match.group(0), "")
        print(f"count_penalty: {count_penalty}")
    if ap_match:
        presence_penalty = float(ap_match.group(2))
        message = message.replace(ap_match.group(0), "")
        print(f"presence_penalty: {presence_penalty}")

    temp = clamp(temp, 0.2, 5)
    top_p = max(0, top_p)
    count_penalty = clamp(count_penalty, 0.0, 1.0)
    presence_penalty = clamp(presence_penalty, 0.0, 1.0)
    return message, temp, top_p, count_penalty, presence_penalty


def translate_message(message, from_lang, to_lang):
    translator = translate.Translator(to_lang, from_lang)
    translated = translator.translate(message)
    if from_lang == "autodetect":
        translated = message if translated == SAME_LANG else translated
    elif from_lang != to_lang:
        print(f"translated from {from_lang}: {translated}")
    return translated


def on_reset(user: User) -> str:
    out, model_state, model_tokens = load_all_state("", "chat_intro")
    save_all_state(user.id, "chat", out, model_state, model_tokens)
    save_arguments(user.id, "chat",
                   mode="Casual",
                   temp=1.0,
                   top_p=0.7,
                   count_penalty=0.2,
                   presence_penalty=0.2)

    reply = f"Chat reset for {user.nickname}."
    return reply


def on_reset_bot(user: User) -> str:
    out, model_state, model_tokens = load_all_state("", "chat_intro_bot")
    save_all_state(user.id, "chat", out, model_state, model_tokens)
    save_arguments(user.id, "chat",
                   mode="Assistant",
                   temp=0.8,
                   top_p=0.5,
                   count_penalty=0.1,
                   presence_penalty=0.1)

    reply = f"Chat reset for {user.nickname}. Bot context loaded."
    return reply


def on_translate(user: User, message: str) -> str:
    lang_match = re.search("\-([a-z]{2}(-[A-Z]{2})?)\s+", message)
    to_lang = "zh"

    if lang_match is not None:
        message = message.replace(lang_match.group(0), "")
        to_lang = lang_match.group(1)

    from_lang = langid.classify(message)[0]
    reply = translate_message(message, from_lang, to_lang)
    reply = f"Translated from {from_lang} to {to_lang}:\n{reply}"
    return reply


def on_generate(user: User, message: str, mode: str = "") -> str:
    message = message.replace("\r\n", '\n').replace('\\n', '\n').strip()
    if len(message) > MAX_MESSAGE_LEN:
        return f"Your message is too long! (max {MAX_MESSAGE_LEN} tokens)"
    if len(message) == 0:
        return ""
    print(message)

    if mode != "retry" and mode != "more":
        save_arguments(user.id, "gen", mode=mode)

    if mode == "inst":
        temp = 0.8
        top_p = 0.5
        count_penalty = 0.1
        presence_penalty = 0.1
    else:
        temp = 1.0
        top_p = 0.8
        count_penalty = 0.2
        presence_penalty = 0.2

    message, temp, top_p, count_penalty, presence_penalty = read_sampler_params(
        message, temp, top_p, count_penalty, presence_penalty)

    reply: str = ""

    if mode == "retry":
        try:
            out, model_state, model_tokens = load_all_state(user.id, "gen_0")
        except:
            return reply
    elif mode == "more":
        try:
            out, model_state, model_tokens = load_all_state(user.id, "gen_1")
            save_all_state(user.id, "gen_0", out, model_state, model_tokens)
        except:
            return reply
    elif mode == "qa":
        message = user.qa_format(message)
        model_tokens = tokenizer.encode(message)
        out, model_state = run_rnn(model_tokens)
        save_all_state(user.id, "gen_0", out, model_state, model_tokens)
    elif mode == "inst":
        message = user.instruct_format(message)
        model_tokens = tokenizer.encode(message)
        out, model_state = run_rnn(model_tokens)
        save_all_state(user.id, "gen_0", out, model_state, model_tokens)
    else:
        message = '\n' + message.strip()
        model_tokens = tokenizer.encode(message)
        out, model_state = run_rnn(model_tokens)
        save_all_state(user.id, "gen_0", out, model_state, model_tokens)

    active_mode = load_arguments(user.id, "gen")['mode']
    occurrence = {}
    start_time = time.time()

    begin = len(model_tokens)
    end = begin
    for i in range(MAX_GENERATE_LEN):
        if active_mode != "qa" and active_mode != "inst":
            out[0] = DONT_OUTPUT
        for n in occurrence:
            out[n] -= presence_penalty + occurrence[n] * count_penalty

        token = tokenizer.sample_logits(out, temp, top_p)
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        model_tokens += [token]
        out, model_state = run_rnn([token], model_state)

        xxx = tokenizer.decode(model_tokens[end:])
        if '\ufffd' not in xxx:
            print(xxx, end='', flush=True)
            end = begin + i + 1

        reply = tokenizer.decode(model_tokens[begin:])
        reply = reply.replace("\r\n", '\n').replace('\\n', '\n')

        if token == 0:
            break

    end_time = time.time()
    delta_time = end_time - start_time
    print(f"\nTokens: {end - begin}\nTime: {delta_time}")

    clear_cache()
    save_all_state(user.id, "gen_1", out, model_state, model_tokens)

    reply = reply.strip()
    return reply


def on_message(user: User, message: str, alt: bool = False) -> str:
    message = message.replace('\r\n', '\n').replace('\\n', '\n').strip()
    message = re.sub("\n(\s*\n)+", '\n', message)

    if len(message) > MAX_MESSAGE_LEN:
        return f"Your message is too long! (max {MAX_MESSAGE_LEN} tokens)"
    if len(message) == 0:
        return ""
    print(message)

    # lang = langid.classify(message)[0]
    reply: str = ""

    try:
        channel = "chat_previous" if alt else "chat"
        out, model_state, model_tokens = load_all_state(user.id, channel)

        arguments = load_arguments(user.id, "chat")
        mode = arguments['mode']
        temp = arguments['temp']
        top_p = arguments['top_p']
        count_penalty = arguments['count_penalty']
        presence_penalty = arguments['presence_penalty']

        message, temp, top_p, count_penalty, presence_penalty = \
            read_sampler_params(message,
                                temp,
                                top_p,
                                count_penalty,
                                presence_penalty)
        save_arguments(user.id, "chat",
                       mode=mode,
                       temp=temp,
                       top_p=top_p,
                       count_penalty=count_penalty,
                       presence_penalty=presence_penalty)
    except:
        intro = "chat_intro"
        out, model_state, model_tokens = load_all_state("", intro)
        save_all_state(user.id, "chat", out, model_state, model_tokens)

        mode = "Casual"
        temp = 1.0
        top_p = 0.7
        count_penalty = 0.2
        presence_penalty = 0.2
        save_arguments(user.id, "chat",
                       mode=mode,
                       temp=temp,
                       top_p=top_p,
                       count_penalty=count_penalty,
                       presence_penalty=presence_penalty)

        if alt:
            return reply

    print(f'''Temperature: {temp}
Top p: {top_p}
Count Penalty: {count_penalty}
Presence Penalty: {presence_penalty}''')
    print(f"{user.bot_name}{user.interface}", end='')

    if not alt:
        if mode == "Assistant":
            message = user.chat_format(message, 'Bob', 'Alice')
        else:
            message = user.chat_format(message)
        tokens = tokenizer.encode(message)

        model_tokens += tokens
        out, model_state = run_rnn(tokens, model_state)

        save_all_state(
            user.id,
            "chat_previous",
            out,
            model_state,
            model_tokens)

    occurrence = {}
    begin = len(model_tokens)
    end = begin
    for i in range(MAX_REPLY_LEN):
        if i <= 0:
            nl_bias = DONT_OUTPUT
        elif i <= 30:
            nl_bias = (i - 30) * 0.1
        else:
            nl_bias = 0
        # else:
        #     nl_bias = (i - 300) * 0.25
        out[187] += nl_bias
        for n in occurrence:
            out[n] -= presence_penalty + occurrence[n] * count_penalty

        token = tokenizer.sample_logits(out, temp, top_p)
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        tokens = tokenizer.encode('\n\n') if token == 0 else [token]
        model_tokens += tokens
        out, model_state = run_rnn(tokens, model_state)

        xxx = tokenizer.decode(model_tokens[end:])
        if '\ufffd' not in xxx:
            print(xxx, end='', flush=True)
            end = begin + i + 1

        reply = tokenizer.decode(model_tokens[begin:])
        reply = reply.replace("\r\n", '\n').replace('\\n', '\n')

        if '\n\n' in reply:
            break

        # State recovery
        def recover_state(forbidden: str, reply: str, out, model_state, model_tokens):
            idx = reply.find(forbidden)
            if idx < 0:
                return idx, reply, out, model_state, model_tokens

            reply = f" {reply[:idx].strip()}\n\n"
            tokens = tokenizer.encode(reply)
            out, model_state, model_tokens = \
                load_all_state(user.id, "chat_previous")

            model_tokens += tokens
            out, model_state = run_rnn(tokens, model_state)

            return idx, reply, out, model_state, model_tokens

        idx, reply, out, model_state, model_tokens = recover_state(
            f"{user.name}{user.interface}",
            reply,
            out,
            model_state,
            model_tokens)
        if idx >= 0:
            print(f"\nRecovered: {tokenizer.decode(model_tokens[begin:])}")
            break

        idx, reply, out, model_state, model_tokens = recover_state(
            f"{user.bot_name}{user.interface}",
            reply,
            out,
            model_state,
            model_tokens)
        if idx >= 0:
            print(f"\nRecovered: {tokenizer.decode(model_tokens[begin:])}")
            break

    clear_cache()
    save_all_state(user.id, "chat", out, model_state, model_tokens)

    reply = reply.replace(user.name, user.nickname)
    reply = reply.replace(user.name.lower(), user.nickname)
    reply = reply.replace(user.name.upper(), user.nickname)
    reply = reply.strip()
    # reply = translate_message(reply, "en", lang)
    return reply


if __name__ == "__main__":
    init_run()
