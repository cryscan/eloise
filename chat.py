import os
import copy
import sys
import types
import gc
import numpy as np
import torch

from model.model_run import RWKV_RNN
from model.utils import TOKENIZER

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
except:
    pass

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
np.set_printoptions(precision=4, suppress=True, linewidth=200)

CHAT_LANG = 'English'  # English Chinese
# CHAT_LANG = 'Chinese'

WORD_NAME = [
    "20B_tokenizer.json",
    "20B_tokenizer.json",
]  # [vocab, vocab] for Pile model
UNKNOWN_CHAR = None
tokenizer = TOKENIZER(WORD_NAME, UNKNOWN_CHAR=UNKNOWN_CHAR)

NO_NEWLINE = -999999999
MAX_REPLY_LEN = 1024

args = types.SimpleNamespace()
args.RUN_DEVICE = "cuda"  # 'cpu' (already very fast) // 'cuda'
# fp32 (good for CPU) // fp16 (recommended for GPU) // bf16 (less accurate)
args.FLOAT_MODE = "fp16"
args.vocab_size = 50277
args.head_qk = 0
args.pre_ffn = 0
args.grad_cp = 0
args.my_pos_emb = 0

args.MODEL_NAME = '/root/autodl-tmp/Models/RWKV-4-Pile-7B-20221115-8047'
args.n_layer = 32
args.n_embd = 4096
args.ctx_len = 1024

model = None


def load_model():
    # Load Model
    global model
    print(f"Loading... {args.MODEL_NAME}")

    os.environ["RWKV_RUN_DEVICE"] = args.RUN_DEVICE
    model = RWKV_RNN(args)


model_tokens = []
current_state = None


def run_rnn(tokens, nl_bias=0):
    global model_tokens, current_state
    for i in range(len(tokens)):
        model_tokens += [int(tokens[i])]
        if i == len(tokens) - 1:
            out, current_state = model.forward(model_tokens, current_state)
        else:
            current_state = model.forward(
                model_tokens, current_state, preprocess_only=True)

    out[0] = NO_NEWLINE
    out[187] += nl_bias

    return out


all_state = {}


def save_all_state(uid, channel, last_out):
    global all_state
    n = f'{uid}_{channel}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(current_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)


def load_all_state(uid, channel):
    global all_state, model_tokens, current_state
    n = f'{uid}_{channel}'
    current_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']


# Profile


# user = "User"
# bot = "Bot"
# interface = ":"

# chatbot_intro = f'''
# The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

# {user}{interface} french revolution what year

# {bot}{interface} The French Revolution started in 1789, and lasted 10 years until 1799.

# {user}{interface} 3+5=?

# {bot}{interface} The answer is 8.

# {user}{interface} guess i marry who ?

# {bot}{interface} Only if you tell me more about yourself - what are your interests?

# {user}{interface} solve for a: 9-a=2

# {bot}{interface} The answer is a = 7, because 9 - 7 = 2.

# {user}{interface} wat is lhc

# {bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

# '''

user = "Simmons"
bot = "Lucy"
pron = "her"
interface = ":"

chatbot_intro = f'''
The following is a verbose detailed conversation between a boy {user} and a young girl {bot}. {bot} really likes role playing. {bot} is intelligent, friendly and cute. {bot} always tells everything {pron} knows to {user}.

{user}{interface} Hello {bot}, are you going to school today?

{bot}{interface} No, today is holiday.

{user}{interface} That's nice! Do you want to chat with me a while?

{bot}{interface} Of course! I'm listening.

'''


def init_run():
    out = run_rnn(tokenizer.tokenizer.encode(chatbot_intro))
    gc.collect()
    torch.cuda.empty_cache()

    save_all_state("", "chatbot_intro", out)


def clamp(n, minimum, maximum):
    return max(minimum, min(n, maximum))


def read_sampler_params(msg):
    x_temp = 1.0
    x_top_p = 0.85
    if ("-temp=" in msg):
        x_temp = float(msg.split("-temp=")[1].split(" ")[0])
        msg = msg.replace("-temp="+f'{x_temp:g}', "")
        # print(f"temp: {x_temp}")
    if ("-top_p=" in msg):
        x_top_p = float(msg.split("-top_p=")[1].split(" ")[0])
        msg = msg.replace("-top_p="+f'{x_top_p:g}', "")
        # print(f"top_p: {x_top_p}")

    x_temp = clamp(x_temp, 0.2, 5)
    x_top_p = max(0, x_top_p)
    return x_temp, x_top_p


def on_reset(uid, nickname) -> str:
    out = load_all_state("", "chatbot_intro")
    save_all_state(uid, "chat", out)
    return f"Chat reset for {nickname}."


def on_generate(uid, message: str, mode: str = "") -> str:
    global model_tokens, current_state

    msg = message.replace('\r\n', '\n').replace('\\n', '\n').strip()
    if len(msg) > 1024:
        return "Your message is too long! (max 1024 tokens)"

    x_temp, x_top_p = read_sampler_params(msg)

    if mode == "retry":
        try:
            out = load_all_state(uid, "gen_0")
        except:
            return ""
    elif mode == "more":
        try:
            out = load_all_state(uid, "gen_1")
            save_all_state(uid, "gen_0", out)
        except:
            return ""
    else:
        next = '\n' + message.strip()
        if mode == "qa":
            next = f"\nQuestion: {message.strip()}\n\nFull Expert Answer:\n"

        current_state = None
        out = run_rnn(tokenizer.tokenizer.encode(next))
        save_all_state(uid, "gen_0", out)

    reply = ""

    begin = len(model_tokens)
    out_last = begin
    for i in range(150):
        if i <= 0:
            nl_bias = NO_NEWLINE
        elif i <= 30:
            nl_bias = (i - 30) * 0.1
        elif i <= 130:
            nl_bias = 0
        else:
            nl_bias = (i - 130) * 0.25
        token = tokenizer.sample_logits(
            out,
            model_tokens,
            args.ctx_len,
            temperature=x_temp,
            top_p_usual=x_top_p,
            top_p_newline=x_top_p,
        )
        out = run_rnn([token], nl_bias=nl_bias)

        xxx = tokenizer.tokenizer.decode(model_tokens[out_last:])
        if '\ufffd' not in xxx:
            out_last = begin + i + 1

        reply = tokenizer.tokenizer.decode(model_tokens[begin:])
        reply = reply.replace('\r\n', '\n').replace('\\n', '\n')
        if '\n\n' in reply:
            reply = reply.replace('\n\n', '\n').strip()
            break
    print('\n')

    reply = tokenizer.tokenizer.decode(model_tokens[begin:]).strip()
    reply = reply.replace('\r\n', '\n').replace('\\n', '\n').replace('\n\n', '\n')

    save_all_state(uid, "gen_1", out)
    return reply


def on_message(uid, message: str, alt: bool = False) -> str:
    global model_tokens, current_state

    msg = message.replace('\r\n', '\n').replace('\\n', '\n').strip()
    if len(msg) > 1024:
        return "Your message is too long! (max 1024 tokens)"

    x_temp, x_top_p = read_sampler_params(msg)

    if not alt:
        try:
            out = load_all_state(uid, "chat")
        except:
            out = load_all_state("", "chatbot_intro")
            save_all_state(uid, "chat", out)
        next = f"{user}{interface} {msg}\n\n{bot}{interface}"

        out = run_rnn(tokenizer.tokenizer.encode(next), nl_bias=NO_NEWLINE)
        save_all_state(uid, "chat_previous", out)
    else:
        try:
            out = load_all_state(uid, "chat_previous")
        except:
            return ""

    reply = ""
    begin = len(model_tokens)
    out_last = begin
    for i in range(MAX_REPLY_LEN):
        if i <= 0:
            nl_bias = NO_NEWLINE
        elif i <= 30:
            nl_bias = (i - 30) * 0.1
        elif i <= 130:
            nl_bias = 0
        else:
            nl_bias = (i - 130) * 0.25
        token = tokenizer.sample_logits(
            out,
            model_tokens,
            args.ctx_len,
            temperature=x_temp,
            top_p_usual=x_top_p,
            top_p_newline=x_top_p
        )
        out = run_rnn([token], nl_bias=nl_bias)

        xxx = tokenizer.tokenizer.decode(model_tokens[out_last:])
        if '\ufffd' not in xxx:
            # print(xxx, end='', flush=True)
            out_last = begin + i + 1

        reply = tokenizer.tokenizer.decode(model_tokens[begin:])
        reply = reply.replace('\r\n', '\n').replace('\\n', '\n')
        if '\n\n' in reply:
            reply = reply.replace('\n\n', '\n').strip()
            break

    save_all_state(uid, "chat", out)

    return reply
