import os
import copy
import sys
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

DONT_OUTPUT = -999999999
MAX_REPLY_LEN = 1024
AVOID_REPEAT = '，。：？！'

MAX_MESSAGE_LEN = 4096
CHUNK_LEN = 256

args = types.SimpleNamespace()

# args.strategy = 'cpu fp32'
# args.strategy = 'cuda fp16'
# args.strategy = 'cuda fp16 *8 -> cpu fp32'
# args.strategy = 'cuda fp16 *6+'
# args.strategy = 'cuda fp16 *0+ -> cpu fp32 *1'
args.strategy = 'cuda fp16 *33 -> cpu fp32'

# args.MODEL_NAME = '/root/autodl-tmp/Models/RWKV-4-Pile-7B-20221115-8047'
# args.MODEL_NAME = '/root/autodl-tmp/Models/RWKV-4-Pile-14B-20230213-8019'
# args.MODEL_NAME = '/root/autodl-tmp/Models/RWKV-4-Pile-14B-20230228-ctx4096-test663'
args.MODEL_NAME = '/root/autodl-tmp/Models/RWKV-4-Pile-14B-20230313-ctx8192-test1050'

args.STATE_DUMP_NAME = './state_8k'

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


def run_rnn(tokens, nl_bias=0):
    global model_tokens, model_state

    tokens = [int(x) for x in tokens]
    model_tokens += tokens

    while len(tokens) > 0:
        out, model_state = model.forward(tokens[:CHUNK_LEN], model_state)
        tokens = tokens[CHUNK_LEN:]

    out[0] = DONT_OUTPUT
    out[187] += nl_bias

    return out


all_state = {}


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
        print("Loading chat intro male...")
        clear_current_state()
        out = run_rnn(tokenizer.encode(prompt.default_male.chat_intro()))
        save_all_state("", "chat_intro_male", out)
        save_all_state("", "chat_intro_unknown", out)

        print("Loading chat intro female...")
        clear_current_state()
        out = run_rnn(tokenizer.encode(prompt.default_female.chat_intro()))
        save_all_state("", "chat_intro_female", out)

        print("Loading instruct intro...")
        clear_current_state()
        out = run_rnn(tokenizer.encode(prompt.default_male.instruct_intro()))
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
    x_temp = temp
    x_top_p = top_p
    x_af = af
    x_ap = ap

    temp_match = re.search("(\-temp\s*=\s*)([^\s]+)\s+", message)
    top_p_match = re.search("(\-top_p\s*=\s*)([^\s]+)\s+", message)
    af_match = re.search("(\-af\s*=\s*)([^\s]+)\s+", message)
    ap_match = re.search("(\-ap\s*=\s*)([^\s]+)\s+", message)

    if temp_match is not None:
        x_temp = float(temp_match.group(2))
        message = message.replace(temp_match.group(0), "")
        print(f"temp: {x_temp}")
    if top_p_match is not None:
        x_top_p = float(top_p_match.group(2))
        message = message.replace(top_p_match.group(0), "")
        print(f"top_p: {x_top_p}")
    if af_match is not None:
        x_af = float(af_match.group(2))
        message = message.replace(af_match.group(0), "")
        print(f"af: {x_af}")
    if ap_match is not None:
        x_ap = float(ap_match.group(2))
        message = message.replace(ap_match.group(0), "")
        print(f"ap: {x_ap}")

    x_temp = clamp(x_temp, 0.2, 5)
    x_top_p = max(0, x_top_p)
    x_af = clamp(x_af, 0.0, 1.0)
    x_ap = clamp(x_ap, 0.0, 1.0)
    return message, x_temp, x_top_p, x_af, x_ap


def translate_message(message, from_lang, to_lang):
    translator = translate.Translator(to_lang, from_lang)
    translated = translator.translate(message)
    translated = message if translated == SAME_LANG else translated
    print(f"translated: {translated}")
    return translated


def on_reset(user: User) -> str:
    out = load_all_state("", f"chat_intro_{user.sex}")
    reply = f"Chat reset for {user.nickname}."
    save_all_state(user.id, "chat", out)
    return reply


def on_generate(user: User, message: str, mode: str = "") -> str:
    global model_tokens, model_state

    message = message.replace("\r\n", '\n').replace('\\n', '\n').strip()
    if len(message) > MAX_MESSAGE_LEN:
        return f"Your message is too long! (max {MAX_MESSAGE_LEN} tokens)"
    print(message)

    if mode != "retry" and mode != "more":
        save_active_mode(user.id, "gen", mode)

    x_temp = 1.0
    x_top_p = 0.8
    x_af = 0.5
    x_ap = 0.2
    if mode == "inst":
        x_temp = 0.2
        x_top_p = 0.5
        x_af = 0.1
        x_ap = 0.1

    message, x_temp, x_top_p, x_af, x_ap = read_sampler_params(
        message, x_temp, x_top_p, x_af, x_ap)

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
        next = user.qa_format(message)
        out = run_rnn(tokenizer.encode(next))
        save_all_state(user.id, "gen_0", out)
    elif mode == "inst":
        clear_current_state()
        out = load_all_state("", f"instruct_intro")
        next = user.instruct_format(message)
        out = run_rnn(tokenizer.encode(next))
        save_all_state(user.id, "gen_0", out)
    else:
        clear_current_state()
        next = '\n' + message.strip()
        out = run_rnn(tokenizer.encode(next))
        save_all_state(user.id, "gen_0", out)

    active_mode = load_active_mode(user.id, "gen")

    counter = torch.zeros_like(out, device=out.device)
    begin = len(model_tokens)
    out_last = begin
    for i in range(150):
        out = tokenizer.alpha_logits(
            out, counter, alpha_frequency=x_af, alpha_presence=x_ap)
        token = tokenizer.sample_logits(out, temperature=x_temp, top_p=x_top_p)
        out = run_rnn([token])

        if active_mode == "inst":
            if i <= 0:
                nl_bias = DONT_OUTPUT
            elif i <= 30:
                nl_bias = (i - 30) * 0.1
            elif i <= 130:
                nl_bias = 0
            else:
                nl_bias = (i - 130) * 0.25

            # Suppress the output of "###"
            out[4118] += nl_bias
            out[817] += nl_bias
            out[4] += nl_bias

        counter[int(token)] += 1

        xxx = tokenizer.decode(model_tokens[out_last:])
        if '\ufffd' not in xxx:
            print(xxx, end='', flush=True)
            out_last = begin + i + 1

        reply = tokenizer.decode(model_tokens[begin:])
        reply = reply.replace("\r\n", '\n').replace('\\n', '\n')

        if active_mode == "qa" and '\n\n' in reply:
            break
        elif active_mode == "inst" and '''###---''' in reply:
            reply = reply[:-6]
            break

    save_all_state(user.id, "gen_1", out)
    return reply


def on_message(user: User, message: str, alt: bool = False) -> str:
    global model_tokens, model_state

    message = message.replace('\r\n', '\n').replace('\\n', '\n').strip()
    if len(message) > MAX_MESSAGE_LEN:
        return f"Your message is too long! (max {MAX_MESSAGE_LEN} tokens)"
    print(message)

    message, x_temp, x_top_p, x_af, x_ap = read_sampler_params(message)
    reply: str = ""

    src_lang = langid.classify(message)[0]
    message = translate_message(message, src_lang, "en")

    if not alt:
        try:
            out = load_all_state(user.id, "chat")
        except:
            out = load_all_state("", f"chat_intro_{user.sex}")
            save_all_state(user.id, "chat", out)
        next = user.chat_format(message)

        out = run_rnn(tokenizer.encode(next), nl_bias=DONT_OUTPUT)
        save_all_state(user.id, "chat_previous", out)
    else:
        try:
            out = load_all_state(user.id, "chat_previous")
        except:
            return reply

    counter = torch.zeros_like(out, device=out.device)
    begin = len(model_tokens)
    out_last = begin
    for i in range(MAX_REPLY_LEN):
        if i <= 0:
            nl_bias = DONT_OUTPUT
        elif i <= 30:
            nl_bias = (i - 30) * 0.1
        elif i <= 130:
            nl_bias = 0
        else:
            nl_bias = (i - 130) * 0.25

        out = tokenizer.alpha_logits(
            out, counter, alpha_frequency=x_af, alpha_presence=x_ap)
        token = tokenizer.sample_logits(out, temperature=x_temp, top_p=x_top_p)
        out = run_rnn([token], nl_bias=nl_bias)
        counter[int(token)] += 1

        xxx = tokenizer.decode(model_tokens[out_last:])
        if '\ufffd' not in xxx:
            print(xxx, end='', flush=True)
            out_last = begin + i + 1

        reply = tokenizer.decode(model_tokens[begin:])
        reply = reply.replace("\r\n", '\n').replace('\\n', '\n')

        if '\n\n' in reply:
            break

    save_all_state(user.id, "chat", out)

    reply = reply.replace(user.name(), user.nickname)
    reply = reply.replace(user.name().lower(), user.nickname)
    reply = reply.replace(user.name().upper(), user.nickname)
    reply = reply.strip()
    # reply = translate_message(reply, "en", src_lang)
    return reply


if __name__ == "__main__":
    init_run()
