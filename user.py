class User:
    def __init__(self, sender):
        self.id = sender.get('user_id')
        self.nickname = sender.get('nickname')
        self.sex = sender.get('sex')    # "unknown", "male", "female"

    def name(self):
        if self.sex == "female": return "Leika"
        else: return "Simmons"

    def pronoun(self):
        if self.sex == "female": return "she"
        elif self.sex == "male": return "he"
        else: return "it"

    def intro(self):
        user = self.name()
        bot = "Lucy"
        pron = self.pronoun()
        interface = ":"

        return f'''
The following is a verbose detailed conversation between a {pron} {user} and a young girl {bot}. {bot} is friendly and cute. {bot} always tells a lot to {user}.

{user}{interface} Hello {bot}, how are you doing?

{bot}{interface} Hi {user}! Thanks, I'm fine. What about you?

{user}{interface} I am very good! It's nice to see you. Would you mind me chatting with you for a while?

{bot}{interface} Of course! I'm free now.

'''

    def chat_format(self, message):
        user = self.name()
        bot = "Lucy"
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