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


def save_all_state(srv, name, last_out):
    global all_state
    n = f'{srv}_{name}'
    all_state[n] = {}
    all_state[n]['out'] = last_out
    all_state[n]['rnn'] = copy.deepcopy(current_state)
    all_state[n]['token'] = copy.deepcopy(model_tokens)


def load_all_state(srv, name):
    global all_state, model_tokens, current_state
    n = f'{srv}_{name}'
    current_state = copy.deepcopy(all_state[n]['rnn'])
    model_tokens = copy.deepcopy(all_state[n]['token'])
    return all_state[n]['out']


# Profile


user = "User"
bot = "Bot"
interface = ":"

chatbot_intro = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} french revolution what year

{bot}{interface} The French Revolution started in 1789, and lasted 10 years until 1799.

{user}{interface} 3+5=?

{bot}{interface} The answer is 8.

{user}{interface} guess i marry who ?

{bot}{interface} Only if you tell me more about yourself - what are your interests?

{user}{interface} solve for a: 9-a=2

{bot}{interface} The answer is a = 7, because 9 - 7 = 2.

{user}{interface} wat is lhc

{bot}{interface} LHC is a high-energy particle collider, built by CERN, and completed in 2008. They used it to confirm the existence of the Higgs boson in 2012.

'''


def init_run():
    out = run_rnn(tokenizer.tokenizer.encode(chatbot_intro))
    gc.collect()
    torch.cuda.empty_cache()

    save_all_state("", "chatbot_intro", out)


def clamp(n, minimum, maximum):
    return max(minimum, min(n, maximum))


def init_chat_state(srv):
    try:
        load_all_state(srv, "chat")
    except:
        on_reset(srv)


def on_reset(srv) -> str:
    out = load_all_state("", "chatbot_intro")
    save_all_state(srv, "chat", out)
    return "Chat reset."


def on_message(srv, message: str, retry: bool) -> str:
    global model_tokens, current_state

    msg = message.replace('\\n', '\n').strip()
    if len(msg) > 2048:
        return "Your message is too long! (max 2048 tokens)"

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

    if not retry:
            init_chat_state(srv)
            out = load_all_state(srv, "chat")
            next_in = f"{user}{interface} {msg}\n\n{bot}{interface}"

            out = run_rnn(tokenizer.tokenizer.encode(next_in), nl_bias=NO_NEWLINE)
            save_all_state(srv, "chat_previous", out)
    else:
        try:
            out = load_all_state(srv, "chat_previous")
        except:
            return ""

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

        send_msg = tokenizer.tokenizer.decode(model_tokens[begin:])
        if '\n\n' in send_msg:
            send_msg = send_msg.strip()
            return send_msg

    return ""
