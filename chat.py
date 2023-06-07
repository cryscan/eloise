import os
import copy
import sys
from enum import Enum
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
from model.utils import TOKENIZER, SAMPLER
from prompt import User, Scenario, SCENARIO_ALICE, SCENARIO_ELOISE, SCENARIO_NEURO

import prompt

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

DONT_OUTPUT = -float('inf')
END_OF_TEXT = 0
END_OF_LINE = 187
END_OF_LINE_DOUBLE = 535
END_OF_LINE_DOUBLE_TRIE = 261

MAX_MESSAGE_LEN = 8192
CHUNK_LEN = 256

MAX_GENERATE_LEN = 250
MAX_REPLY_LEN = 1024

# CHAT_SAMPLER = SAMPLER("typical", 1.0, 0.8, 0.4, 0.1, 0.1, 256)
CHAT_SAMPLER = SAMPLER("nucleus", 1.0, 0.7, 0.4, 0.1, 0.1, 256)
INSTRUCT_SAMPLER = SAMPLER("nucleus", 1.0, 0.5, 0.95, 0.3, 0.3, 256)

args = types.SimpleNamespace()

# tokenizer = TOKENIZER("20B_tokenizer.json")
tokenizer = TOKENIZER("rwkv_vocab_v20230424")

# args.strategy = 'cpu fp32'
args.strategy = 'cuda fp16'
# args.strategy = 'cuda fp16 *8 -> cpu fp32'
# args.strategy = 'cuda fp16 *6+'
# args.strategy = 'cuda fp16 *0+ -> cpu fp32 *1'
# args.strategy = 'cuda fp16 *32 -> cpu fp32'
# args.strategy = 'cuda fp16 *20 -> cpu fp32'
# args.strategy = 'cuda fp16i8 *16 -> cuda fp16'

args.MODEL_NAME = '/root/autodl-tmp/models/RWKV-4-World-7B-v1-OnlyForTest_52%_trained-20230606-ctx4096'
# args.MODEL_NAME = '/root/autodl-tmp/models/RWKV-4-Raven-14B-v12-Eng98%-Other2%-20230523-ctx8192'
# args.MODEL_NAME = '/root/autodl-tmp/models/RWKV-4-Raven-7B-v11-Eng49%-Chn49%-Jpn1%-Other1%-20230430-ctx8192'

args.STATE_DUMP_NAME = 'states/14b.state'
# args.STATE_DUMP_NAME = 'states/7b.state'


class GenerateMode(Enum):
    GENERATE = 0
    INSTRUCT = 1
    RETRY = 2
    MORE = 3


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


def state_to_cuda(state):
    if state:
        for i in range(model.args.n_layer):
            dd = model.strategy[i]
            dev = dd.device
            state[i*5+0] = state[i*5+0].to(dev)
            state[i*5+1] = state[i*5+1].to(dev)
            state[i*5+2] = state[i*5+2].to(dev)
            state[i*5+3] = state[i*5+3].to(dev)
            state[i*5+4] = state[i*5+4].to(dev)


def state_to_cpu(state):
    if state:
        for i in range(model.args.n_layer):
            state[i*5+0] = state[i*5+0].cpu()
            state[i*5+1] = state[i*5+1].cpu()
            state[i*5+2] = state[i*5+2].cpu()
            state[i*5+3] = state[i*5+3].cpu()
            state[i*5+4] = state[i*5+4].cpu()


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
    state_to_cpu(all_state[n]['state'])


def load_all_state(uid, channel):
    n = f'{uid}_{channel}'
    model_state = copy.deepcopy(all_state[n]['state'])
    model_tokens = copy.deepcopy(all_state[n]['token'])

    state_to_cuda(model_state)
    return all_state[n]['out'], model_state, model_tokens


def save_params(uid, channel, **kwargs):
    n = f'params_{uid}_{channel}'
    all_state[n] = kwargs


def load_params(uid, channel):
    n = f'params_{uid}_{channel}'
    return all_state[n]


def clear_cache():
    gc.collect()
    torch.cuda.empty_cache()


def fix_tokens_end_line(tokens):
    if not tokenizer.is_trie() and tokens and tokens[-1] == END_OF_LINE_DOUBLE:
        tokens = tokens[:-1] + [END_OF_LINE, END_OF_LINE]
        # print("Tokens fixed")
    return tokens


def fix_tokens_end_text(tokens):
    if tokens and tokens[-1] == END_OF_TEXT:
        if tokenizer.is_trie():
            tokens = tokens[:-1] + [END_OF_LINE_DOUBLE_TRIE]
        else:
            tokens = tokens[:-1] + [END_OF_LINE, END_OF_LINE]
    return tokens


def init_run():
    # try:
    #     recover_all_state()
    #     print("Recovered state")
    # except:
    print("Loading chat intro...")
    scenario = SCENARIO_ELOISE
    tokens = tokenizer.encode(scenario.intro())
    tokens = fix_tokens_end_line(tokens)
    out, state = run_rnn(tokens)
    save_all_state("", scenario.intro.__name__, out, state, tokens)

    print("Loading chat intro...")
    scenario = SCENARIO_ALICE
    tokens = tokenizer.encode(scenario.intro())
    tokens = fix_tokens_end_line(tokens)
    out, state = run_rnn(tokens)
    save_all_state("", scenario.intro.__name__, out, state, tokens)

    print("Loading chat intro...")
    scenario = SCENARIO_NEURO
    tokens = tokenizer.encode(scenario.intro())
    tokens = fix_tokens_end_line(tokens)
    out, state = run_rnn(tokens)
    save_all_state("", scenario.intro.__name__, out, state, tokens)

    clear_cache()
    # dump_all_state()


def recover_all_state():
    global all_state
    with open(args.STATE_DUMP_NAME, 'rb') as file:
        all_state = pickle.load(file)


def dump_all_state():
    with open(args.STATE_DUMP_NAME, 'wb') as file:
        pickle.dump(all_state, file, protocol=pickle.HIGHEST_PROTOCOL)


def clamp(n, minimum, maximum):
    return max(minimum, min(n, maximum))


def translate_message(message, from_lang, to_lang):
    translator = translate.Translator(to_lang, from_lang)
    translated = translator.translate(message)
    if from_lang == "autodetect":
        translated = message if translated == SAME_LANG else translated
    elif from_lang != to_lang:
        print(f"translated from {from_lang}: {translated}")
    return translated


def on_reset(user: User, message: str, scenario: Scenario, sampler: SAMPLER) -> str:
    out, model_state, model_tokens = load_all_state(
        '', scenario.intro.__name__)
    scenario = copy.deepcopy(scenario)
    sampler = copy.deepcopy(sampler)
    message = sampler.parse(message)

    save_all_state(user.id, "chat", out, model_state, model_tokens)
    save_params(user.id, "chat", scenario=scenario, sampler=sampler)

    return f"Chat reset for {user.nickname}. You are {scenario.user_name} and I am {scenario.bot_name}."


def on_show_params(user: User, message: str) -> str:
    try:
        params = load_params(user.id, "chat")
        scenario: Scenario = params['scenario']
        sampler: SAMPLER = params['sampler']
        message = sampler.parse(message)
        save_params(user.id, "chat", scenario=scenario, sampler=sampler)
    except:
        sampler = copy.deepcopy(CHAT_SAMPLER)
        scenario = copy.deepcopy(SCENARIO_ELOISE)
        message = sampler.parse(message)
        save_params(user.id, "chat", scenario=scenario, sampler=sampler)
    return str(sampler)


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


def on_generate(user: User, message: str, mode=GenerateMode.GENERATE) -> str:
    message = message.replace("\r\n", '\n').replace('\\n', '\n').strip()
    if len(message) > MAX_MESSAGE_LEN:
        return f"Your message is too long! (max {MAX_MESSAGE_LEN} tokens)"
    print(f"{user.nickname}({user.id}): {message}")

    reply: str = ""

    if mode not in [GenerateMode.RETRY, GenerateMode.MORE]:
        if mode == GenerateMode.GENERATE:
            sampler = copy.deepcopy(CHAT_SAMPLER)
        elif mode == GenerateMode.INSTRUCT:
            sampler = copy.deepcopy(INSTRUCT_SAMPLER)

        message = sampler.parse(message)
        active_mode = mode
        save_params(user.id, "gen", mode=mode, sampler=sampler)
    else:
        try:
            params = load_params(user.id, "gen")
            sampler: SAMPLER = params['sampler']
            active_mode = params['mode']

            message = sampler.parse(message)
            save_params(user.id, "gen", mode=active_mode, sampler=sampler)
        except Exception as e:
            print(e)
            return reply

    print(str(sampler))

    if mode == GenerateMode.RETRY:
        try:
            out, model_state, model_tokens = load_all_state(user.id, "gen_0")
        except:
            return reply
    elif mode == GenerateMode.MORE:
        try:
            out, model_state, model_tokens = load_all_state(user.id, "gen_1")
            save_all_state(user.id, "gen_0", out, model_state, model_tokens)
        except:
            return reply
    elif mode == GenerateMode.INSTRUCT:
        message = prompt.instruct_format(message)
        model_tokens = tokenizer.encode(message)
        out, model_state = run_rnn(model_tokens)
        save_all_state(user.id, "gen_0", out, model_state, model_tokens)
    else:
        message = '\n' + message.strip()
        model_tokens = tokenizer.encode(message)
        out, model_state = run_rnn(model_tokens)
        save_all_state(user.id, "gen_0", out, model_state, model_tokens)

    occurrence = {}
    start_time = time.time()

    begin = len(model_tokens)
    end = begin
    for i in range(MAX_GENERATE_LEN):
        if active_mode == GenerateMode.GENERATE:
            out[0] = DONT_OUTPUT
        for n in occurrence:
            out[n] -= sampler.presence_penalty + \
                occurrence[n] * sampler.count_penalty

        token = sampler.sample(out)
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        if i > sampler.penalty_range:
            return_token = model_tokens[-sampler.penalty_range]
            if return_token in occurrence:
                occurrence[return_token] -= 1
                if occurrence[return_token] == 0:
                    del occurrence[return_token]

        if token != END_OF_TEXT:
            model_tokens += [token]
        out, model_state = run_rnn([token], model_state)

        xxx = tokenizer.decode(model_tokens[end:])
        if '\ufffd' not in xxx:
            print(xxx, end='', flush=True)
            end = begin + i + 1

        reply = tokenizer.decode(model_tokens[begin:])
        reply = reply.replace("\r\n", '\n').replace('\\n', '\n')

        if token == END_OF_TEXT:
            break

    end_time = time.time()
    delta_time = end_time - start_time
    print(f"\nTokens: {end - begin}\nTime: {delta_time}")

    clear_cache()
    save_all_state(user.id, "gen_1", out, model_state, model_tokens)

    reply = reply.strip()
    return reply


def on_message(user: User, message: str, alt=False) -> str:
    message = message.replace('\r\n', '\n').replace('\\n', '\n').strip()
    message = re.sub("\n(\s*\n)+", '\n', message)

    if len(message) > MAX_MESSAGE_LEN:
        return f"Your message is too long! (max {MAX_MESSAGE_LEN} tokens)"
    if not alt and len(message) == 0:
        return ""
    print(f"{user.nickname}({user.id}): {message}")

    # lang = langid.classify(message)[0]
    reply: str = ""

    try:
        channel = "chat_pre" if alt else "chat"
        out, model_state, model_tokens = load_all_state(user.id, channel)

        params = load_params(user.id, "chat")
        scenario: Scenario = params['scenario']
        sampler: SAMPLER = params['sampler']
        message = sampler.parse(message)
        save_params(user.id, "chat", scenario=scenario, sampler=sampler)
    except:
        if alt:
            return reply

        scenario = copy.deepcopy(SCENARIO_ELOISE)
        sampler = copy.deepcopy(CHAT_SAMPLER)
        message = sampler.parse(message)

        out, model_state, model_tokens = load_all_state(
            '', scenario.intro.__name__)

        save_all_state(user.id, "chat", out, model_state, model_tokens)
        save_params(user.id, "chat", scenario=scenario, sampler=sampler)

    print(str(sampler))
    print(f"{scenario.bot_name}{scenario.interface}", end='')

    if not alt:
        message = scenario.chat_format(message)
        tokens = tokenizer.encode(message)

        model_tokens += tokens
        out, model_state = run_rnn(tokens, model_state)

        save_all_state(
            user.id,
            "chat_pre",
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
        out[END_OF_LINE] += nl_bias
        for n in occurrence:
            out[n] -= sampler.presence_penalty + \
                occurrence[n] * sampler.count_penalty

        token = sampler.sample(out)
        if token != END_OF_LINE:
            if token not in occurrence:
                occurrence[token] = 1
            else:
                occurrence[token] += 1

        if i > sampler.penalty_range:
            return_token = model_tokens[-sampler.penalty_range]
            if return_token in occurrence:
                occurrence[return_token] -= 1
                if occurrence[return_token] == 0:
                    del occurrence[return_token]

        tokens = fix_tokens_end_text([token])
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
            tokens = fix_tokens_end_line(tokens)
            out, model_state, model_tokens = \
                load_all_state(user.id, "chat_pre")

            model_tokens += tokens
            out, model_state = run_rnn(tokens, model_state)

            return idx, reply, out, model_state, model_tokens

        idx, reply, out, model_state, model_tokens = recover_state(
            f"{scenario.user_name}{scenario.interface}",
            reply,
            out,
            model_state,
            model_tokens)
        if idx >= 0:
            print(f"\nRecovered: {tokenizer.decode(model_tokens[begin:])}")
            break

        idx, reply, out, model_state, model_tokens = recover_state(
            f"{scenario.bot_name}{scenario.interface}",
            reply,
            out,
            model_state,
            model_tokens)
        if idx >= 0:
            print(f"\nRecovered: {tokenizer.decode(model_tokens[begin:])}")
            break

    clear_cache()
    save_all_state(user.id, "chat", out, model_state, model_tokens)

    reply = reply.replace(scenario.user_name, user.nickname)
    reply = reply.replace(scenario.user_name.lower(), user.nickname)
    reply = reply.replace(scenario.user_name.upper(), user.nickname)
    reply = reply.strip()
    # reply = translate_message(reply, "en", lang)
    return reply


if __name__ == "__main__":
    init_run()
