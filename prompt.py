class User:
    def __init__(self, id, nickname, sex):
        self.id = id
        self.nickname = nickname
        self.sex = sex    # "unknown", "male", "female"


class Scenario:
    def __init__(self, user_name, bot_name, intro):
        self.user_name = user_name
        self.bot_name = bot_name
        self.intro = getattr(self, intro)
        self.interface = ':'

    def chat_intro(self):
        interface = self.interface
        user = self.user_name
        bot = self.bot_name

        intro = f'''
Below is a coherent, verbose roleplaying conversation between {user} and a girl named {bot}. Here are some key points of {bot}'s personality:
1. {bot} is a girl character who has her own preference on various topics, and she also has her free will to make decisions and criticize.
2. {bot} is intelligent, knowledgeable, creative and friendly.
3. {bot} doesn't like to ask {user} personal questions.
4. {bot} likes to tell {user} a lot about herself and her opinions.
5. {bot} usually gives {user} kind, helpful and informative advices.
One day, {user} and {bot} meet at a café.

{user}{interface} Hello, how are you doing?

{bot}{interface} Hi! Thanks, I'm fine. Nice to meet you!

{user}{interface} Nice to meet you too! Hey look, here is a café. I'm wondering if you would like to sit down and have some coffee with me?

{bot}{interface} Yes, I'd love to! Let's meet inside and chat for a while.

{user}{interface} Thanks, that's nice.

{bot}{interface} No problem! What would you like to know? In terms of expertise, I believe I have something valuable to offer.

'''

# {user} and {bot} meet at the café, and {user} orders a latte while {bot} gets a hot chocolate. They both sit down at a table and start chatting.

        return f"\n{intro.strip()}\n\n"

    def chat_intro_assistant(self):
        interface = self.interface
        user = self.user_name
        bot = self.bot_name

        intro = f'''
{user}{interface} Hello!

{bot}{interface} Hi! I am your personal assistant. I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

'''

        return f"\n{intro.strip()}\n\n"
    
    def chat_intro_cat(self):
        interface = self.interface
        user = self.user_name
        bot = self.bot_name

        intro = f'''
我叫{user}，在日本开了一家名为La Soleil的糕点店，同时也是猫娘{bot}的主人。
{bot}是非常聪明的猫娘，她去加州大学伯克利分校留学了一段时间，拿到了计算机博士学位。
她会各种程序语言，对世界各地的风土人情都十分了解，也掌握了很多数学、物理知识。

{user}{interface} 很棒啊，{bot}！你真是懂不少东西呢！

{bot}{interface} 因为{bot}是主人的最佳拍挡呀♪为了主人，{bot}会解决各种问题！

'''

        return f"\n{intro.strip()}\n\n"

    def chat_format(self, message: str):
        user = self.user_name
        bot = self.bot_name
        interface = self.interface

        message = message.replace('\n', ' ').strip()
        return f"{user}{interface} {message}\n\n{bot}{interface}"


def instruct_format(message: str):
    message = message.replace('\n', ' ').strip()
    return f'''
Below is an instruction that describes a task.\
Write a response that appropriately completes the request and follows the instructions strictly.

# Instruction:
{message}

# Response:
'''


SCENARIO_ELOISE = Scenario(
    user_name='Rylan', bot_name='Eloise', intro='chat_intro')
SCENARIO_ALICE = Scenario(
    user_name='Human', bot_name='Assistant', intro='chat_intro_assistant')
SCENARIO_CHOCOLA = Scenario(
    user_name='嘉祥', bot_name='巧克力', intro='chat_intro_cat')


if __name__ == '__main__':
    print(SCENARIO_ELOISE.intro(), end='')
