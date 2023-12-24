import random
import json
import pickle
import time
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import temp

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json', encoding='UTF-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# temp.py 모듈화된 함수 호출
def get_weather_response(city):
    return temp.get_weather_data(city)

# clean_up_sentence 및 bag_of_words 함수 추가
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# predict_class 함수 추가
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# get_response 함수 수정
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = None

    # 추가된 부분: "날씨"와 관련된 패턴 처리
    if tag == '날씨':
        city = input("어디알고싶은데? ")  # 사용자로부터 도시 입력 받기
        result = get_weather_response(city)
    else:
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break

    if result is None:
        result = "무슨말을 하는지 모르겠어 ㅠㅠ"

    return result

print("여자친구 깨우기 성공")

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
