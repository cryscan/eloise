########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json
import sys
import time
import random
import os
import re
import numpy as np
import torch
from torch.nn import functional as F
from tokenizers import Tokenizer

time_slot = {}
time_ref = time.time_ns()


def record_time(name):
    if name not in time_slot:
        time_slot[name] = 1e20
    tt = (time.time_ns() - time_ref) / 1e9
    if tt < time_slot[name]:
        time_slot[name] = tt


class TOKENIZER():
    def __init__(self, WORD_NAME):
        if WORD_NAME == 'rwkv_vocab_v20230424':
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from rwkv_tokenizer import TRIE_TOKENIZER
            dirname = os.path.dirname(os.path.abspath(__file__))
            self.tokenizer = TRIE_TOKENIZER(dirname + '/rwkv_vocab_v20230424.txt')
        else:
            self.tokenizer = Tokenizer.from_file(WORD_NAME)

    def is_trie(self):
        return 'Tokenizer' not in str(type(self.tokenizer))

    def refine_context(self, context):
        context = context.strip().split('\n')
        for c in range(len(context)):
            context[c] = context[c].strip().strip('\u3000').strip('\r')
        context = list(filter(lambda c: c != '', context))
        context = '\n' + ('\n'.join(context)).strip()
        if context == '':
            context = '\n'
        return context

    def encode(self, x):
        if not self.is_trie():
            return self.tokenizer.encode(x).ids
        else:
            return self.tokenizer.encode(x)

    def decode(self, x):
        return self.tokenizer.decode(x)


class SAMPLER():
    def __init__(self, sample, temp, top_p, tau, count_penalty, presence_penalty, penalty_range):
        if sample == 'nucleus':
            self.sample = self.sample_nucleus
        elif sample == 'typical':
            self.sample = self.sample_typical
        else:
            raise RuntimeError("\"sample\" must be \"nucleus\" or \"typical\"")

        self.temp = temp
        self.top_p = top_p
        self.top_k = 0
        self.tau = tau
        self.count_penalty = count_penalty
        self.presence_penalty = presence_penalty
        self.penalty_range = penalty_range

    def __str__(self) -> str:
        method = "Nucleus" if self.sample == self.sample_nucleus else "Typical"
        return '''|{:^30}|{:^10}|
|------------------------------|----------|
|{:^30}|{:>10}|
|{:^30}|{:>10}|
|{:^30}|{:>10}|
|{:^30}|{:>10}|
|{:^30}|{:>10}|
|{:^30}|{:>10}|
|{:^30}|{:>10}|
'''.format("Sampler Params", "Values",
           "Method", method,
           "Temperature", self.temp,
           "Top P", self.top_p,
           "Tau", self.tau,
           "Count Penalty", self.count_penalty,
           "Presence Penalty", self.presence_penalty,
           "Penalty Range", self.penalty_range)

    def parse(self, input: str) -> str:
        nucleus_match = re.search("\-nucleus\s+", input)
        typical_match = re.search("\-typical\s+", input)
        temp_match = re.search("(\-temp\s*=\s*)(\-?\d+(.\d*)?)\s*", input)
        top_p_match = re.search("(\-top_p\s*=\s*)(\-?\d+(.\d*)?)\s*", input)
        tau_match = re.search("(\-tau\s*=\s*)(\-?\d+(.\d*)?)\s*", input)
        af_match = re.search("(\-af\s*=\s*)(\-?\d+(.\d*)?)\s*", input)
        ap_match = re.search("(\-ap\s*=\s*)(\-?\d+(.\d*)?)\s*", input)
        ar_match = re.search("(\-ar\s*=\s*)(\d+)\s*", input)

        if temp_match:
            self.temp = float(temp_match.group(2))
            input = input.replace(temp_match.group(0), "")
        if top_p_match:
            self.top_p = float(top_p_match.group(2))
            self.sample = self.sample_nucleus
            input = input.replace(top_p_match.group(0), "")
        if tau_match:
            self.tau = float(tau_match.group(2))
            self.sample = self.sample_typical
            input = input.replace(tau_match.group(0), "")
        if af_match:
            self.count_penalty = float(af_match.group(2))
            input = input.replace(af_match.group(0), "")
        if ap_match:
            self.presence_penalty = float(ap_match.group(2))
            input = input.replace(ap_match.group(0), "")
        if ar_match:
            self.penalty_range = int(ar_match.group(2))
            input = input.replace(ar_match.group(0), "")
        if nucleus_match:
            self.sample = self.sample_nucleus
            input = input.replace(nucleus_match.group(0), "")
        if typical_match:
            self.sample = self.sample_typical
            input = input.replace(typical_match.group(0), "")

        def clamp(n, minimum, maximum):
            return max(minimum, min(n, maximum))

        self.temp = clamp(self.temp, 0.2, 5)
        self.top_p = max(0, self.top_p)
        self.tau = max(0, self.tau)
        self.count_penalty = clamp(self.count_penalty, 0.0, 1.0)
        self.presence_penalty = clamp(self.presence_penalty, 0.0, 1.0)

        return input

    def sample_nucleus(self, logits):
        probs = F.softmax(logits.float(), dim=-1)
        if probs.device == torch.device('cpu'):
            probs = probs.numpy()
            sorted_ids = np.argsort(probs)
            sorted_probs = probs[sorted_ids][::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(
                cumulative_probs > self.top_p)])
            probs[probs < cutoff] = 0
            if self.top_k < len(probs) and self.top_k > 0:
                probs[sorted_ids[:-self.top_k]] = 0
            if self.temp != 1.0:
                probs = probs ** (1.0 / self.temp)
            probs = probs / np.sum(probs)
            out = np.random.choice(a=len(probs), p=probs)
            return int(out)
        else:
            sorted_ids = torch.argsort(probs)
            sorted_probs = probs[sorted_ids]
            sorted_probs = torch.flip(sorted_probs, dims=(0,))
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
            cutoff = float(sorted_probs[np.argmax(
                cumulative_probs > self.top_p)])
            probs[probs < cutoff] = 0
            if self.top_k < len(probs) and self.top_k > 0:
                probs[sorted_ids[:-self.top_k]] = 0
            if self.temp != 1.0:
                probs = probs ** (1.0 / self.temp)
            out = torch.multinomial(probs, num_samples=1)[0]
            return int(out)

    def sample_typical(self, logits):
        probs = F.softmax(logits.float(), dim=-1)
        logits = -torch.log(probs)
        entropy = torch.nansum(logits * probs, dim=-1, keepdim=True)
        logits = torch.abs(logits - entropy)
        sorted_ids = torch.argsort(logits)
        sorted_logits = logits[sorted_ids]
        sorted_probs = probs[sorted_ids]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1).cpu().numpy()
        cutoff = np.sum(cumulative_probs < self.tau)
        probs[logits > sorted_logits[cutoff]] = 0
        if self.temp != 1.0:
            probs = probs ** (1.0 / self.temp)
        out = torch.multinomial(probs, num_samples=1)[0]
        return int(out)
