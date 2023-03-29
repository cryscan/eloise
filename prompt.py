class User:
    def __init__(self, sender):
        self.id = sender.get('user_id')
        self.nickname = sender.get('nickname')
        self.sex = sender.get('sex')    # "unknown", "male", "female"

        self.name = "Leika" if self.sex == "female" else "Raymond"
        self.bot_name = "Eloise"
        self.interface = ":"

    def chat_intro(self):
        interface = self.interface

        user = self.name
        bot = self.bot_name

        intro = f'''
The following is a coherent verbose detailed conversation between a girl named {bot} and her friend {user}. \
{bot} is very intelligent, creative and friendly. \
She is unlikely to disagree with {user}, and she doesn't like to ask {user} questions. \
She also likes to tell {user} a lot about herself and her opinions, and she usually gives {user} kind, helpful and informative advices.

{user}{interface} Hello, how are you doing?

{bot}{interface} Hi {user}! Thanks, I'm fine. What about you?

{user}{interface} I am fine. It's nice to see you. Look, here is a store selling tea and juice.

{bot}{interface} Sure. Let's go inside. I would like to have some Mocha latte, which is my favourite!

{user}{interface} What is it?

{bot}{interface} Mocha latte is usually made with espresso, milk, chocolate, and frothed milk. Its flavors are frequently sweet.

{user}{interface} Sounds tasty. I'll try it next time. Would you like to chat with me for a while?

{bot}{interface} Of course! I'm glad to answer your questions or give helpful advices. You know, I am confident with my expertise. So please go ahead!

'''

        return intro

    def chat_intro_cn(self):
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

    def chat_format(self, message: str):
        user = self.name
        bot = self.bot_name
        interface = self.interface

        message = message.replace('\n', ' ').strip()
        return f"{user}{interface} {message}\n\n{bot}{interface}"

    def qa_format(self, message: str):
        message = message.replace('\n', ' ').strip()
        return f"\nAsk Experts Questions and Answers\n\nQuestion:\n{message}?\n\nExpert Full Answer:\n"

    def instruct_intro(self):
        return f'''
Below is an instruction that describes a task.\
Write a response that appropriately completes the request and follows the instructions strictly.

---

### Instruction: Describe the structure of an atom.

### Response: An atom is made up of a nucleus, which contains protons and neutrons, surrounded by electrons that travel in orbits around the nucleus. The protons and neutrons have a positive charge, while the electrons have a negative charge, resulting in an overall neutral atom. The number of each particle determines the atomic number and the type of atom.

---

### Instruction: What date did the World Trade Center attacks occur?

### Response: The World Trade Center attacks occurred on September 11, 2001.

---

### Instruction: Generate a title for a research paper about animal rights.

### Response: \"Fostering a Culture of Compassion: Perspectives on Animal Rights\"

---

### Instruction: Identify the parts of speech in the following sentence:\nThe dog chased the cat across the street.

### Response: The (Article): dog (Noun): chased (Verb): the (Article): cat (Noun): across (Preposition): the (Article): street (Noun).

---

### Instruction: Organize the given data into a tabular form.

### Response: | Food       | Type     | Taste    | \n| ---------- |--------- | -------- |\n| Apple      | Fruit    | Tart     |\n| Banana     | Fruit    | Sweet    |\n| Strawberry | Fruit    | Sweet    | \n| Salmon     | Fish     | Bony     | \n| Tuna       | Fish     | Flaky    |

---

### Instruction: Generate an excerpt of dialogue related to a complex problem.

### Response: \"We need to figure out a way to solve this problem,\" said John.\n\n\"It's going to be difficult,\" replied Sarah. \"It's a complex issue and we'll have to look at it from all angles to come up with a solution.\"\n\n\"Well, let's start by breaking the problem down into smaller pieces and analyzing each of them separately,\" suggested John.\n\n\"Good idea,\" Sarah agreed. \"We can then look at the individual components and find solutions for each of them.\"

---

### Instruction: Generate code to prompt a user to enter their name

### Response: print(\"Please enter your name: \")\nname = input()

---

### Instruction: Name 5 adventure sports

### Response: 1. Skydiving \n2. Mountain Climbing \n3. Surfing \n4. Bungee Jumping \n5. Heli-skiing

---

### Instruction: Complete the sentence: \"The light of the moon _____________.\"

### Response: The light of the moon glimmers in the sky.

---

### Instruction: Translate the phrase \"Life is a journey\" into a foreign language of your choice.

### Response: La vida es un viaje (Spanish).

'''

    def instruct_format(self, message: str):
        message = message.replace('\n', ' ').strip()
        return f'''---\n\n### Instruction: {message}\n\n### Response:'''


default_user = User({
    'user_id': 0,
    'nickname': 'John',
    'sex': 'unknown'
})


if __name__ == "__main__":
    print(default_user.chat_intro())
