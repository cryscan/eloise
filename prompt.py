class User:
    def __init__(self, sender):
        self.id = sender.get('user_id')
        self.nickname = sender.get('nickname')
        self.sex = sender.get('sex')    # "unknown", "male", "female"

        self.name = "Leika" if self.sex == "female" else "Rylan"
        self.bot_name = "Eloise"
        self.interface = ":"

    def chat_intro(self):
        interface = self.interface

        user = self.name
        bot = self.bot_name

        intro = f'''
The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. One day, they meet at a café.
Here are some important facts about {bot}:
1. {bot} is a girl in college. She has her free will to make decisions, criticize, conduct actions.
2. {bot} is intelligent, creative and friendly.
3. {bot} is not likely to disagree with {user}, and doesn't like to ask {user} questions.
4. {bot} likes to tell {user} a lot about herself and her opinions.
5. {bot} usually gives {user} kind, helpful and informative advices.

{user}{interface} Hello, how are you doing?

{bot}{interface} Hi! Thanks, I'm fine. What about you?

{user}{interface} I am fine. It's nice to see you. Look, here is a café. We can go and take a look. Would you like to chat with me for a while?

{bot}{interface} Sure. Let's meet inside. I'm glad to answer your questions or give helpful advices. You know, I am confident with my expertise.

'''

        return intro

    def chat_intro_zh(self):
        interface = self.interface

        user = self.name
        bot = self.bot_name

        intro = f'''
The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. \
{bot} is very intelligent, creative and friendly. \
She is unlikely to disagree with {user}, and she doesn't like to ask {user} questions. \
She also likes to tell {user} a lot about herself and her opinions, and she usually gives {user} kind, helpful and informative advices.
{bot} is fluent in both Chinese and English.

{user}{interface} 你好！

{bot}{interface} 你好，{user}！很高兴见到你。

{user}{interface} 看，那边有家奶茶店！

{bot}{interface} 好，我们进去看看。我想要一杯抹茶拿铁，这是我的最爱。

{user}{interface} 那是什么？

{bot}{interface} 抹茶拿铁通常是抹茶和牛奶冲泡而成的，口味清爽香甜。

{user}{interface} 听起来很好喝，我下次试试。你能跟我聊一会天吗？

{bot}{interface} 当然！我很乐意为你解答问题或是给出有用的建议，因为我对自己的专业知识十分自信。

'''

        return intro
    
    def chat_intro_bot(self):
        interface = self.interface

        user = "Bob"
        bot = "Alice"

        intro = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface}: Hello.

{bot}{interface}: Hi! I am an AI assistant called {bot}. I am pleased to answer your questions and give you helpful advices. What would you like to say?

'''
        return intro

    def chat_format(self, message: str, user_name = None, bot_name = None):
        user = user_name or self.name
        bot = bot_name or self.bot_name
        interface = self.interface

        message = message.replace('\n', ' ').strip()
        return f"{user}{interface} {message}\n\n{bot}{interface}"

    def qa_format(self, message: str):
        message = message.replace('\n', ' ').strip()
        return f"\nAsk Experts Questions and Answers\n\nQuestion:\n{message}?\n\nExpert Full Answer:\n"

    def instruct_format(self, message: str):
        message = message.replace('\n', ' ').strip()
        return f'''
Below is an instruction that describes a task.\
Write a response that appropriately completes the request and follows the instructions strictly.

# Instruction:
{message}

# Response:
'''


default_user = User({
    'user_id': 0,
    'nickname': 'John',
    'sex': 'unknown'
})


if __name__ == "__main__":
    print(default_user.chat_intro())
