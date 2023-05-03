# Eloise
A QQ Chatbot based on RWKV (W.I.P.)

## Introduction
This is a bot for QQ IM software based on the [Language Model of RWKV](https://github.com/BlinkDL/RWKV-LM).

## Run
1. Install [`go-cqhttp`](https://docs.go-cqhttp.org/).
2. Edit `config.yml`; fill in your QQ and password.
3. Edit `chat.py`; change your model path.
4. Create 3 empty folders in the project path: `logs`, `images` and `states`.
5. Open two terminals.
   ```bash
   cd /path/to/eloise
   ```
6. Run `go-cqhttp` in one terminal for the first time; follow instructions.
7. Edit `device.json`; change `protocol` to `2`.
8. Run `go-cqhttp` again; follow instructions to log in.
9. Run `./run.sh` in another terminal.
