# ConvBot_group

Robotics Club Summer Project 2020
* Team Members:
  * Abhay Dayal Mathur    - https://github.com/Stellarator-X
  * Amitesh Singh Sisodia - https://github.com/Amitesh163
  * Anchal Gupta          - https://github.com/anchalgupta05
  * Arpit Verma           - https://github.com/Av-hash
  * Manit Ajmera          - https://github.com/manitajmera
  * Sanskar Mittal        - https://github.com/sanskarm
  


***
## Aim
The aim of this project was to make a **Talking bot** , one which can pay attention to the user's voice and make a meaningful and contextual response according to their intent, just like we're talking to a human.

***
## Ideation 
This project was divided into overall three parts :

* [Speech to Text conversion](https://github.com/Amitesh163/ConvBot_group/tree/master/SpeechRecognition)
* [Response Generation](https://github.com/Amitesh163/ConvBot_group/tree/master/Response%20Generation)
* [Text to speech conversion](https://github.com/Amitesh163/ConvBot_group/tree/master/TextToSpeech)

***
## Usage

Installing required dependencies : `$pip install requirements.txt`

Training checkpoints, LDA model weights and tokens can be found [here](https://drive.google.com/drive/folders/18o-bFpJjy1S4UHUbdTjQnb2B_IK4bIM5?usp=sharing)

Required File Structure:

```txt
ResponseGen + AIML
├── AIML files
│   └── ...
├── bot
│   ├── LDA
│   ├── Tokens.txt
│   ├── topic_dict.dict
│   └── training_checkpoints
└── ...
```

### Running the bot

```bash
usage: bot.py [-h] [-m {msg,trigger}]

The bot.

optional arguments:
  -h, --help            show this help message and exit
  -m {msg,trigger}, --mode {msg,trigger}
                        Mode of execution : Message box/ Trigger word
                        detection
```

### Functionality

* Casual Conversations
* Google Search
* Weather Information

***
### Demonstration
The video demonstration of this project can be found [here](https://drive.google.com/file/d/1jAmxwfUnrx9qa9nh8Sol4ZByIH_w7YRE/view?usp=drivesdk).

***
## References

* _Deep Speech 2: End-to-End Speech Recognition in English and Mandarin_
  * **Link** : [https://arxiv.org/abs/1512.02595]
  * **Author(s)/Organization** : Baidu Research – Silicon Valley AI Lab
  * **Tags** : Speech Recognition
  * **Published** : 8 Dec, 2015

* _Topic Aware Neural Response Generation_
  * **Link** : [https://arxiv.org/abs/1606.08340]
  * **Authors** : Chen Xing, Wei Wu, Yu Wu, Jie Liu, Yalou Huang, Ming Zhou, Wei-Ying Ma
  * **Tags** : Neural response generation; Sequence to sequence model; Topic aware conversation model; Joint attention; Biased response generation
  * **Published** : 21 Jun 2016 (v1), 19 Sep 2016 (v2)

* _Topic Modelling and Event Identification from Twitter Textual Data_
  * **Link** : [https://arxiv.org/abs/1608.02519]
  * **Authors** : Marina Sokolova, Kanyi Huang, Stan Matwin, Joshua Ramisch, Vera Sazonova, Renee Black, Chris Orwa, Sidney Ochieng, Nanjira Sambuli
  * **Tags** : Latent Dirichlet Allocation; Topic Models; Statistical machine translation
  * **Published** : 8 Aug 2016

