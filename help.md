# Help
Model: <model>

### Misc Utilities
1. `-tr [-en, -zh] <text>`: Translate
2. `-p, -params`: Show chat parameters

### Free Generation
1. `-h, -help`: Show this help
2. `-g, -gen <text>`: Generate text
3. `-e, -retry`: Retry last generation
4. `-m, -more`: Generate more
6. `-i, -inst <text>`: Follow instructions

### Chat
1. `-s, -reset`: Reset your chat chain (Casual)
2. `-b, -bot`: Reset your chat chain (AI Assistant)
3. `-a, -alt`: Alternative reply
4. <chat>: Chat with me

### Parameters
| Param             | Description                        | Default (Chat Mode) | Default (Bot/Instruction Mode) |
| ----------------- | ---------------------------------- | ------------------- | ------------------------------ |
| `-nucleus`        | Switch to nucleus sampling         | <chat_nucleus>      | <inst_nucleus>                 |
| `-typical`        | Switch to typical sampling         | <chat_typical>      | <inst_typical>                 |
| `-temp=<number>`  | Higher temperature → more verbose  | <chat_temp>         | <inst_temp>                    |
| `-top_p=<number>` | Lower top p → more accurate answer | <chat_top_p>        | <inst_top_p>                   |
| `-tau=<number>`   | Lower tau → more human-like answer | <chat_tau>          | <inst_tau>                     |
| `-af=<number>`    | Count penalty, avoids repeating    | <chat_af>           | <inst_af>                      |
| `-ap=<number>`    | Presence penalty                   | <chat_ap>           | <inst_ap>                      |

### Examples
#### Chat
* Use `-alt` or `-a` to retry
* Use `-reset` or `-s` to reset, better at chillin'
* Use `-bot` or `-b` to enter bot mode, better at QA and coding
* Do not use `-retry` or `-more`: useless

```text
<user>: What's your interesting news today?

<bot>: ...

<user>: -alt

<bot>: ...

<user>: -reset

<bot>： Chat reset for ...
```

#### Instruct
* Use `-retry` or `-e` to retry
* Use `-more` to continue

```text
<user>: -i Write a letter to our customers, apologizing for delay in delivery of our products.

<bot>: Dear Valued Customers:
...
Yours Sincerely
[YOUR NAME]

<user>: -retry

<bot>: ...
```

#### Translation
```text
<user>: -tr Hello World!

<bot>: 你好，世界！

<user>: -tr -en 你好，世界！

<bot>: Hello, world!
```

#### Generation
```text
<user>: -gen -temp=2.0 -top_p=0.5 Here is a story about a man's lonely journey on Mars:

<bot>: (Some story...)

<user>: -m

<bot>: (Continue generates...)
```