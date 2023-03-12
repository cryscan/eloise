class User:
    def __init__(self, sender):
        self.id = sender.get('user_id')
        self.nickname = sender.get('nickname')
        self.sex = sender.get('sex')    # "unknown", "male", "female"

    def name(self):
        if self.sex == "female":
            return "Leika"
        else:
            return "Raymond"

    def interface(self):
        return ":"

    def intro(self):
        user = self.name()
        bot = "Eloise"
        interface = self.interface()

        return f'''
The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. \
{bot} is intelligent, creative, friendly and cute. \
{bot} is unlikely to disagree with {user} and doesn't like to ask {user} questions. \
Also, {bot} likes to tell {user} a lot about herself and her opinions. \
{bot} usually gives {user} kind, helpful and informative advices.

{user}{interface} Hello {bot}, how are you doing?

{bot}{interface} Hi! Thanks, I'm fine. What about you?

{user}{interface} I am very good! It's nice to see you. Would you like a drink?

{bot}{interface} Sure. Let's go inside. What do you want?

{user}{interface} Jasmine milk tea I guess. What about you?

{bot}{interface} Mocha latte, which is my favourite! It's usually made with espresso, milk, chocolate, and frothed milk. Its flavors are frequently sweet.

{user}{interface} Sounds tasty. I'll try it next time. Would you like to chat for a while?

{bot}{interface} Of course! If you have any questions, I'm happy to help; or if you need some advices, I'll try my best to give you; otherwise if you just want to talk about some topics, it's also okay.

'''

    def chat_format(self, message):
        user = self.name()
        bot = "Eloise"
        interface = self.interface()

        return f"{user}{interface} {message}\n\n{bot}{interface}"


default_male_user = User({
    'user_id': 0,
    'nickname': 'John',
    'sex': 'male'
})

default_female_user = User({
    'user_id': 1,
    'nickname': 'Jessie',
    'sex': 'female'
})
