# Eloise
A QQ Chatbot based on RWKV (W.I.P.)

## Introduction
This is a bot for QQ IM software based on the [Language Model of RWKV](https://github.com/BlinkDL/RWKV-LM).

## Run
1. Install [`go-cqhttp`](https://docs.go-cqhttp.org/).
2. Edit `config.yml`; fill in your QQ and password.
3. Open two terminals.
   ```bash
   cd /path/to/eloise
   ```
4. Run `go-cqhttp` in one terminal for the first time; follow instructions.
5. Edit `device.json`; change `protocol` to `2`.
6. Run `go-cqhttp` again; follow instructions to log in.
7. Run `./run.sh` in another terminal.
