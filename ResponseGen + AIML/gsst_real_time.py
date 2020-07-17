
import speech_recognition as sr
r = sr.Recognizer()

def get_transcript():
    mic = sr.Microphone()
    print("stage-1")
    with mic as source:
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
            try :
                result = r.recognize_google(audio)
            except :
                print("no result")
                return None
        except:
            print("failed")
            return None
    return print(result)
#get_transcript()
