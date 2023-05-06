class User:
    def __init__(self, sender):
        self.id = sender.get('user_id')
        self.nickname = sender.get('nickname')
        self.sex = sender.get('sex')    # "unknown", "male", "female"


class Scenario:
    def __init__(self, user_name, bot_name, intro):
        self.user_name = user_name
        self.bot_name = bot_name
        self.intro = getattr(self, intro)
        self.interface = ':'
        self.end = '\n\n'

    def chat_intro(self):
        interface = self.interface
        end = self.end
        user = self.user_name
        bot = self.bot_name

        intro = f'''
Below is a coherent, verbose roleplaying conversation between {user} a girl named {bot}. Here are some key points of {bot}'s personality:
1. {bot} is a human girl who has her own preference on various topics, and she also has her free will to make decisions and criticize.
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

        return f"\n{intro.strip()}{end}"

    def chat_intro_assistant(self):
        interface = self.interface
        end = self.end
        user = self.user_name
        bot = self.bot_name

        intro = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface} Hello.

{bot}{interface} Hi! I am an AI assistant called {bot}. I am pleased to answer your questions and give you helpful advices. What would you like to say?

'''
        return f"\n{intro.strip()}{end}"

    def chat_format(self, message: str):
        user = self.user_name
        bot = self.bot_name
        interface = self.interface
        end = self.end

        message = message.replace('\n', ' ').strip()
        return f"{user}{interface} {message}{end}{bot}{interface}"


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
    user_name='Bob', bot_name='Alice', intro='chat_intro_assistant')


if __name__ == '__main__':
    print(SCENARIO_ELOISE.intro(), end='')
