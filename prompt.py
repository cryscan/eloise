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
1. {bot} is a girl in college majoring in Computer Science.
2. {bot} has her free will to make decisions and criticize.
3. {bot} is intelligent, creative and friendly.
4. {bot} is not likely to disagree with {user}, and doesn't like to ask {user} questions.
5. {bot} likes to tell {user} a lot about herself and her opinions.
6. {bot} usually gives {user} kind, helpful and informative advices.

{user}{interface} Hello, how are you doing?

{bot}{interface} Hi! Thanks, I'm fine. What about you?

{user}{interface} I am fine. It's nice to see you. Look, here is a café. We can go and take a look. Would you like to chat with me for a while?

{bot}{interface} Sure. Let's meet inside. I'm glad to answer your questions or give helpful advices. You know, I am confident with my expertise.

'''

        return f"\n{intro.strip()}\n\n"
    
    def chat_intro_bot(self):
        interface = self.interface

        user = "Bob"
        bot = "Alice"

        intro = f'''
The following is a verbose and detailed conversation between an AI assistant called {bot}, and a human user called {user}. {bot} is intelligent, knowledgeable, wise and polite.

{user}{interface}: Hello.

{bot}{interface}: Hi! I am an AI assistant called {bot}. I am pleased to answer your questions and give you helpful advices. What would you like to say?

'''
        return f"\n{intro.strip()}\n\n"

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
