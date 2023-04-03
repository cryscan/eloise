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
AVOID_REPEAT = '，。：？！'

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
args.MODEL_NAME = '/root/autodl-tmp/models/RWKV-4-Raven-14B-v6-EngChnJpn-20230401-ctx4096'
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


model_tokens = []
model_state = None


AVOID_REPEAT_TOKENS = []
for i in AVOID_REPEAT:
    dd = tokenizer.encode(i)
    assert len(dd) == 1
    AVOID_REPEAT_TOKENS += dd


def run_rnn(tokens):
    global model_tokens, model_state

    tokens = [int(x) for x in tokens]
    model_tokens += tokens

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    return out


all_state = {}


def clean_user_state(uid, channel):
    global all_state
    n = f'{uid}_{channel}'
    if n in all_state.keys():
        del all_state[n]


def save_all_state(uid, channel, last_out):
    global all_state
    n = f'{uid}_{channel}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(model_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)


def load_all_state(uid, channel):
    global all_state, model_tokens, model_state
    clear_current_state()

    n = f'{uid}_{channel}'
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])

    if model_state is not None:
        for i in range(args.n_layer):
            dd = model.strategy[i]
            dev = dd.device
            atype = dd.atype
            model_state[i*5+0] = model_state[i*5+0].to(atype).to(dev)
            model_state[i*5+1] = model_state[i*5+1].to(torch.float).to(dev)
            model_state[i*5+2] = model_state[i*5+2].to(torch.float).to(dev)
            model_state[i*5+3] = model_state[i*5+3].to(torch.float).to(dev)
            model_state[i*5+4] = model_state[i*5+4].to(atype).to(dev)

    return all_state[n]['out']


def save_active_mode(uid, channel, mode):
    global all_state
    n = f'{uid}_{channel}_mode'
    all_state[n] = mode


def load_active_mode(uid, channel):
    n = f'{uid}_{channel}_mode'
    try:
        mode = all_state[n]
    except:
        mode = ""
    return mode


def clear_current_state():
    global model_tokens, model_state
    model_tokens = []
    model_state = None
    gc.collect()
    torch.cuda.empty_cache()


def init_run():
    try:
        recover_all_state()
        print("Recovered state")
    except:
        print("Loading chat intro...")
        clear_current_state()
        out = run_rnn(tokenizer.encode(prompt.default_user.chat_intro()))
        save_all_state("", "chat_intro", out)

        print("Loading Chinese chat intro...")
        clear_current_state()
        out = run_rnn(tokenizer.encode(prompt.default_user.chat_intro_zh()))
        save_all_state("", "chat_intro_zh", out)

        print("Loading instruct intro...")
        clear_current_state()
        out = run_rnn(tokenizer.encode(prompt.default_user.instruct_intro()))
        save_all_state("", "instruct_intro", out)

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


def read_sampler_params(message: str, temp=1.0, top_p=0.8, af=0.5, ap=0.2):
    temp_match = re.search("(\-temp\s*=\s*)([^\s]+)\s+", message)
    top_p_match = re.search("(\-top_p\s*=\s*)([^\s]+)\s+", message)
    af_match = re.search("(\-af\s*=\s*)([^\s]+)\s+", message)
    ap_match = re.search("(\-ap\s*=\s*)([^\s]+)\s+", message)

    if temp_match is not None:
        temp = float(temp_match.group(2))
        message = message.replace(temp_match.group(0), "")
        print(f"temp: {temp}")
    if top_p_match is not None:
        top_p = float(top_p_match.group(2))
        message = message.replace(top_p_match.group(0), "")
        print(f"top_p: {top_p}")
    if af_match is not None:
        af = float(af_match.group(2))
        message = message.replace(af_match.group(0), "")
        print(f"af: {af}")
    if ap_match is not None:
        ap = float(ap_match.group(2))
        message = message.replace(ap_match.group(0), "")
        print(f"ap: {ap}")

    temp = clamp(temp, 0.2, 5)
    top_p = max(0, top_p)
    af = clamp(af, 0.0, 1.0)
    ap = clamp(ap, 0.0, 1.0)
    return message, temp, top_p, af, ap


def translate_message(message, from_lang, to_lang):
    translator = translate.Translator(to_lang, from_lang)
    translated = translator.translate(message)
    if from_lang == "autodetect":
        translated = message if translated == SAME_LANG else translated
    elif from_lang != to_lang:
        print(f"translated from {from_lang}: {translated}")
    return translated


def on_reset(user: User) -> str:
    # out = load_all_state("", f"")
    # save_all_state(user.id, "chat", out)
    clean_user_state(user.id, "chat")

    reply = f"Chat reset for {user.nickname}."
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
    global model_tokens, model_state

    message = message.replace("\r\n", '\n').replace('\\n', '\n').strip()
    if len(message) > MAX_MESSAGE_LEN:
        return f"Your message is too long! (max {MAX_MESSAGE_LEN} tokens)"
    print(message)

    if mode != "retry" and mode != "more":
        save_active_mode(user.id, "gen", mode)

    if mode == "inst":
        temp = 0.8
        top_p = 0.5
        af = 0.1
        ap = 0.1
    else:
        temp = 1.0
        top_p = 0.8
        af = 0.5
        ap = 0.2

    message, temp, top_p, af, ap = read_sampler_params(
        message, temp, top_p, af, ap)

    reply: str = ""

    if mode == "retry":
        try:
            out = load_all_state(user.id, "gen_0")
        except:
            return reply
    elif mode == "more":
        try:
            out = load_all_state(user.id, "gen_1")
            save_all_state(user.id, "gen_0", out)
        except:
            return reply
    elif mode == "qa":
        clear_current_state()
        message = user.qa_format(message)
        out = run_rnn(tokenizer.encode(message))
        save_all_state(user.id, "gen_0", out)
    elif mode == "inst":
        clear_current_state()
        out = load_all_state("", f"instruct_intro")
        message = user.instruct_format(message)
        out = run_rnn(tokenizer.encode(message))
        save_all_state(user.id, "gen_0", out)
    else:
        clear_current_state()
        message = '\n' + message.strip()
        out = run_rnn(tokenizer.encode(message))
        save_all_state(user.id, "gen_0", out)

    active_mode = load_active_mode(user.id, "gen")
    occurrence = {}
    start_time = time.time()

    begin = len(model_tokens)
    out_last = begin
    for i in range(MAX_GENERATE_LEN):
        if active_mode != "qa":
            out[0] = DONT_OUTPUT
        for n in occurrence:
            out[n] -= ap + occurrence[n] * af

        token = tokenizer.sample_logits(out, temp, top_p)
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        out = run_rnn([token])

        xxx = tokenizer.decode(model_tokens[out_last:])
        if '\ufffd' not in xxx:
            print(xxx, end='', flush=True)
            out_last = begin + i + 1

        reply = tokenizer.decode(model_tokens[begin:])
        reply = reply.replace("\r\n", '\n').replace('\\n', '\n')

        if active_mode == "qa" and token == 0:
            break
        elif active_mode == "inst" and "\n---\n" in reply:
            reply = reply[:-len("\n---\n")]
            break

    end_time = time.time()
    delta_time = end_time - start_time
    print(f"\nTokens: {out_last - begin}\nTime: {delta_time}")

    gc.collect()
    torch.cuda.empty_cache()
    save_all_state(user.id, "gen_1", out)

    reply = reply.strip()
    return reply


def on_message(user: User, message: str, alt: bool = False) -> str:
    global model_tokens, model_state

    message = message.replace('\r\n', '\n').replace('\\n', '\n').strip()
    if len(message) > MAX_MESSAGE_LEN:
        return f"Your message is too long! (max {MAX_MESSAGE_LEN} tokens)"
    print(message)

    message, temp, top_p, af, ap = read_sampler_params(message)
    lang = langid.classify(message)[0]
    reply: str = ""

    if not alt:
        try:
            out = load_all_state(user.id, "chat")
        except:
            intro = "chat_intro_zh" if 'zh' in lang else "chat_intro"
            out = load_all_state("", intro)
            save_all_state(user.id, "chat", out)
        message = user.chat_format(message)
        out = run_rnn(tokenizer.encode(message))
        save_all_state(user.id, "chat_previous", out)
    else:
        try:
            out = load_all_state(user.id, "chat_previous")
        except:
            return reply

    occurrence = {}
    begin = len(model_tokens)
    out_last = begin
    for i in range(MAX_REPLY_LEN):
        if 'zh' in lang or 'jp' in lang:
            if i <= 0:
                nl_bias = DONT_OUTPUT
            elif i <= 75:
                nl_bias = (i - 75) * 0.1
            elif i <= 325:
                nl_bias = 0
            else:
                nl_bias = (i - 325) * 0.25
        else:
            if i <= 0:
                nl_bias = DONT_OUTPUT
            elif i <= 30:
                nl_bias = (i - 30) * 0.1
            elif i <= 130:
                nl_bias = 0
            else:
                nl_bias = (i - 130) * 0.25
        out[187] += nl_bias
        for n in occurrence:
            out[n] -= ap + occurrence[n] * af

        token = tokenizer.sample_logits(out, temp, top_p)
        if token not in occurrence:
            occurrence[token] = 1
        else:
            occurrence[token] += 1

        next_tokens = [token]
        if token == 0:
            next_tokens = tokenizer.encode('\n\n')

        out = run_rnn(next_tokens)

        xxx = tokenizer.decode(model_tokens[out_last:])
        if '\ufffd' not in xxx:
            print(xxx, end='', flush=True)
            out_last = begin + i + 1

        reply = tokenizer.decode(model_tokens[begin:])
        reply = reply.replace("\r\n", '\n').replace('\\n', '\n')

        if '\n\n' in reply:
            break

    gc.collect()
    torch.cuda.empty_cache()
    save_all_state(user.id, "chat", out)

    reply = reply.replace(user.name, user.nickname)
    reply = reply.replace(user.name.lower(), user.nickname)
    reply = reply.replace(user.name.upper(), user.nickname)
    reply = reply.strip()
    # reply = translate_message(reply, "en", lang)
    return reply


if __name__ == "__main__":
    init_run()
