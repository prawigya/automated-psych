from threading import Thread
import Emotion.module_emot as emo
from chatterbot import ChatBot
#from chatterbot.trainers import ListTrainer
from chatterbot.trainers import ChatterBotCorpusTrainer


depresso_bot=ChatBot("DepressionBot",read_only=True ,
                logic_adapters=['chatterbot.logic.BestMatch',
                                {'import_path': 'chatterbot.logic.BestMatch',
                                 'threshold': 0.65,
                                 'default_response' : 'I am sorry, but I can only answer questions related to ACM.'
                                 }
                                ],
                input_adapter = 'chatterbot.input.VariableInputTypeAdapter',
                output_adapter = 'chatterbot.output.OutputAdapter',
                filter = 'chatterbot.filters.RepetitiveResponseFilter'
                )
trainer = ChatterBotCorpusTrainer(depresso_bot)
#trainer = ListTrainer(depresso_bot)
"""
trainer.train(
        ['Hello',
         'Hi',
         'Hello',
         'Okay',
         'Okay :)'
                ]
        )
"""



trainer.train(
    "chatterbot.corpus.english.greetings",
    "chatterbot.corpus.english.conversations",
    "chatterbot.corpus.english.emotion",
    "chatterbot.corpus.english.psychology"
)
emo.emotion_detect(0, depresso_bot)
