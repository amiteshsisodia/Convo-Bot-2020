import sys
sys.path.append("Response Generation/")

import TextToSpeech.tts as tts
import SpeechRecognition.gstt_real_time as stt
import numpy as np
from response_generation import *

done = False

glomar = ['I didn\'t get that', 'You\'ll have to be a little louder.']

while not done:
    print("Say something.")
    stimulus = stt.get_transcript()
    if stimulus is not None:
        if stimulus.lower() == "bye":
            done = True
        print("You said : ", stimulus)
        response = get_response(stimulus)
    else :
        response = np.random.choice(glomar)
    tts.play_response(response)
    print("And I said : ", response)