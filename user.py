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

    def intro(self):
        user = self.name()
        bot = "Eloise"
        interface = ":"

        action = f"[{user} greets {bot}]".capitalize()

        return f'''
The following is a verbose detailed conversation between {user} and a young girl {bot}. {bot} is intelligent, friendly and cute. {bot} is unlikely to disagree with {user}.

{user}{interface} Hello {bot}, how are you doing?

{bot}{interface} Hi! Thanks, I'm fine. What about you?

{user}{interface} I am very good! It's nice to see you. Would you mind me chatting with you for a while?

{bot}{interface} Not at all! I'm listening.

'''

    def chat_format(self, message):
        user = self.name()
        bot = "Eloise"
        interface = ":"

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
