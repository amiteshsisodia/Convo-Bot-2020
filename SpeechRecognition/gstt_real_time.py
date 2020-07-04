import speech_recognition as sr 
r = sr.Recognizer() 

def get_transcript():
    mic = sr.Microphone()
    with mic as source: 
        audio = r.listen(source, timeout=5, phrase_time_limit=10) 
        try :
            result = r.recognize_google(audio)
        except :
            return None 
    return result