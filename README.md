# ConvBot_group

Robotics Club Summer Project

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

