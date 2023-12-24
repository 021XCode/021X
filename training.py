import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from konlpy.tag import Okt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import tensorflow as tf

nltk.download('punkt')
nltk.download('wordnet')

okt = Okt()

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json', encoding='UTF-8').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']


for intent in intents['intents']:
    for pattern in intent['patterns']:
        # 사용자 입력에서 명사 추출 (여기서는 Okt 사용)
        word_list = okt.nouns(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Move the shuffle and conversion outside the loop
random.shuffle(training)
training = np.array(training, dtype=object)

# Rest of the code remains the same
train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# 모델 컴파일 시 해당 옵티마이저를 지정
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)
print("Done")
