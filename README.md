# Eloise
A QQ Chatbot based on RWKV (W.I.P.)

## Introduction
This is a bot for QQ IM software based on the [Language Model of RWKV](https://github.com/BlinkDL/RWKV-LM).

## Run
1. Install [`go-cqhttp`](https://docs.go-cqhttp.org/).
2. Edit `config.yml`; fill in your QQ and password.
3. Check `requirements.txt`; make sure you have all required packages properly installed (you may choose your own torch-gpu version depending on your CUDA version).
4. Edit `chat.py`; change your model path.
5. Create 3 empty folders in the project path: `logs`, `images` and `states`.
6. Open two terminals.
   ```bash
   cd /path/to/eloise
   ```
7. Run `go-cqhttp` in one terminal for the first time; follow instructions.
8. Edit `device.json`; change `protocol` to `2`.
9. Run `go-cqhttp` again; follow instructions to log in.
10. Run `./run.sh` in another terminal.
