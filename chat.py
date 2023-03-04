import os
import copy
import sys
import types
import gc
import numpy as np
import torch
from user import User, default_male_user, default_female_user

from model_v2.model_run import RWKV
from model_v2.utils import TOKENIZER

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

tokenizer = TOKENIZER("20B_tokenizer.json")

DONT_OUTPUT = -999999999
MAX_REPLY_LEN = 1024
AVOID_REPEAT = '，。：？！'

args = types.SimpleNamespace()

# args.strategy = 'cpu fp32'
# args.strategy = 'cuda fp16'
# args.strategy = 'cuda fp16 *8 -> cpu fp32'
# args.strategy = 'cuda fp16 *6+'
# args.strategy = 'cuda fp16 *0+ -> cpu fp32 *1'
args.strategy = 'cuda fp16 *33 -> cpu fp32'

# args.MODEL_NAME = '/root/autodl-tmp/Models/RWKV-4-Pile-7B-20221115-8047'
args.MODEL_NAME = '/root/autodl-tmp/Models/RWKV-4-Pile-14B-20230213-8019'

args.vocab_size = 50277
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0

args.n_layer = 40   # 32
args.n_embd = 5120  # 4096
args.ctx_len = 1024


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
    out, model_state = model.forward(tokens, model_state)

    out[0] = DONT_OUTPUT
    out[187] += nl_bias

    if model_tokens[-1] in AVOID_REPEAT_TOKENS:
        out[model_tokens[-1]] = DONT_OUTPUT

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
    n = f'{uid}_{channel}'
    model_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']


def clear_current_state():
    global model_tokens, model_state
    model_tokens = []
    model_state = None


def init_run():
    clear_current_state()
    out = run_rnn(tokenizer.encode(default_male_user.intro()))

    print("Loading intro male...")
    gc.collect()
    torch.cuda.empty_cache()
    save_all_state("", "intro_male", out)
    save_all_state("", "intro_unknown", out)

    print("Loading intro female...")
    clear_current_state()
    out = run_rnn(tokenizer.encode(default_female_user.intro()))

    gc.collect()
    torch.cuda.empty_cache()
    save_all_state("", "intro_female", out)

    # print("Loading intro male chinese...")
    # clear_current_state()
    # out = run_rnn(tokenizer.encode(default_male_user.intro_cn()))

    # gc.collect()
    # torch.cuda.empty_cache()
    # save_all_state("", "intro_cn_male", out)
    # save_all_state("", "intro_cn_unknown", out)

    # print("Loading intro female chinese...")
    # clear_current_state()
    # out = run_rnn(tokenizer.encode(default_female_user.intro_cn()))

    # gc.collect()
    # torch.cuda.empty_cache()
    # save_all_state("", "intro_cn_female", out)


def clamp(n, minimum, maximum):
    return max(minimum, min(n, maximum))


def read_sampler_params(message):
    x_temp = 1.0
    x_top_p = 0.85
    if ("-temp=" in message):
        x_temp = float(message.split("-temp=")[1].split(" ")[0])
        message = message.replace("-temp="+f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in message):
        x_top_p = float(message.split("-top_p=")[1].split(" ")[0])
        message = message.replace("-top_p="+f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")

    x_temp = clamp(x_temp, 0.2, 5)
    x_top_p = max(0, x_top_p)
    return message, x_temp, x_top_p


def on_reset(user: User, cn: bool = False) -> str:
    out = load_all_state("", f"intro_{user.sex}")
    reply = f"Chat reset for {user.nickname}."
    save_all_state(user.id, "chat", out)
    return reply


def on_generate(user: User, message: str, mode: str = "") -> str:
    global model_tokens, model_state, last_message

    message = message.replace("\r\n", '\n').replace('\\n', '\n').strip()
    if len(message) > 1024:
        return "Your message is too long! (max 1024 tokens)"
    print(message)

    message, x_temp, x_top_p = read_sampler_params(message)

    if mode == "retry":
        try:
            out = load_all_state(user.id, "gen_0")
        except:
            return ""
    elif mode == "more":
        try:
            out = load_all_state(user.id, "gen_1")
            save_all_state(user.id, "gen_0", out)
        except:
            return ""
    else:
        next = '\n' + message.strip()
        if mode == "qa":
            next = f"\nQuestion: {message.strip()}?\n\nExpert Full Answer:\n"

        model_state = None
        out = run_rnn(tokenizer.encode(next))
        save_all_state(user.id, "gen_0", out)

    reply = ""

    counter = torch.zeros_like(out, device=out.device)
    begin = len(model_tokens)
    out_last = begin
    for i in range(150):
        out = tokenizer.alpha_logits(out, counter)
        token = tokenizer.sample_logits(
            out,
            model_tokens,
            args.ctx_len,
            temperature=x_temp,
            top_p=x_top_p
        )
        out = run_rnn([token])
        counter[int(token)] += 1

        xxx = tokenizer.decode(model_tokens[out_last:])
        if '\ufffd' not in xxx:
            print(xxx, end='', flush=True)
            out_last = begin + i + 1

        if mode == "qa":
            reply = tokenizer.decode(model_tokens[begin:])
            reply = reply.replace("\r\n", '\n').replace('\\n', '\n')
            if '\n\n' in reply:
                reply = reply.strip()
                break

    save_all_state(user.id, "gen_1", out)

    reply = tokenizer.decode(model_tokens[begin:]).strip()
    reply = reply.replace("\r\n", '\n')
    reply = reply.replace('\\n', '\n')
    reply = reply.replace("\n\n", '\n')
    return reply.strip()


def on_message(user: User, message: str, alt: bool = False) -> str:
    global model_tokens, model_state

    message = message.replace('\r\n', '\n').replace('\\n', '\n').strip()
    if len(message) > 1024:
        return "Your message is too long! (max 1024 tokens)"
    print(message)

    message, x_temp, x_top_p = read_sampler_params(message)

    if not alt:
        try:
            out = load_all_state(user.id, "chat")
        except:
            out = load_all_state("", f"intro_{user.sex}")
            save_all_state(user.id, "chat", out)
        next = user.chat_format(message)

        out = run_rnn(tokenizer.encode(next), nl_bias=DONT_OUTPUT)
        save_all_state(user.id, "chat_previous", out)
    else:
        try:
            out = load_all_state(user.id, "chat_previous")
        except:
            return ""

    reply = ""

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

        out = tokenizer.alpha_logits(out, counter)
        token = tokenizer.sample_logits(
            out,
            model_tokens,
            args.ctx_len,
            temperature=x_temp,
            top_p=x_top_p
        )
        out = run_rnn([token], nl_bias=nl_bias)
        counter[int(token)] += 1

        xxx = tokenizer.decode(model_tokens[out_last:])
        if '\ufffd' not in xxx:
            print(xxx, end='', flush=True)
            out_last = begin + i + 1

        reply = tokenizer.decode(model_tokens[begin:])
        reply = reply.replace("\r\n", '\n').replace('\\n', '\n')
        if '\n\n' in reply:
            reply = reply.strip()
            break

    save_all_state(user.id, "chat", out)

    reply = reply.replace(user.name(), user.nickname)
    reply = reply.replace(user.name().lower(), user.nickname)
    reply = reply.replace(user.name().upper(), user.nickname)
    return reply
