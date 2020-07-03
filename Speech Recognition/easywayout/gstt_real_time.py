import speech_recognition as sr 
r = sr.Recognizer() 

def get_transcript():
    mic = sr.Microphone(device_index=0)
    with mic as source: 
    audio = r.listen(source) 
    result = r.recognize_google(audio)
    return result