

def binar(digit):  #Бинаризирует число
    string = str(bin(digit))
    digit = string.split('b')[1]
    if len(digit) < 9:
        string = '0' * (9 - len(digit)) + digit
    return string

def get_list(string): #Превращает char->int
    list = []
    for char in string:
        list.append(int(char))
    return list


import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt



test_arr = []
train_num = []
train_labels = []

def sum_list(list1, list2):
    list_return = []
    for item in range(max(len(list1), len(list2))):
        el_now = 0
        try:
            el_now += list1[item]
        except:
            pass
        try:
            el_now += list2[item]
        except:
            pass
        el_now /= 2
        list_return.append(el_now)
    return list_return

print(binar(256))
for item1 in range (102, 150):
    for item2 in range (102, 150):
        test_arr.append([get_list(binar(item1)), get_list(binar(item2))]) #Бинаризирую два числа и создаю массив [item1, item2]


np_test = np.array(test_arr)
for item1 in range (2, 100):
    for item2 in range (2, 100):
        train_num.append([get_list(binar(item1)), get_list(binar(item2))]) #Бинаризирую два числа и создаю массив [item1, item2]
        train_labels.append((get_list(binar(item1 + item2)))) #Результат в бинарном виде

print(train_num)
np_train_numbers = np.array(train_num)
np_train_labels = np.array(train_labels)
print(np_train_numbers)
print(np_train_labels.shape, np_train_numbers.shape)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(2, 9)), #input shape — размерность logits (np_train_numbers)
    keras.layers.Dense(17, activation='relu'),
    keras.layers.Dense(9, activation='relu') #количество битов в числе
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(np_train_numbers, np_train_labels, epochs=500)
test_loss, test_acc = model.evaluate(np_train_numbers,  np_train_labels, verbose=2)
print('\nTest accuracy:', test_acc)
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(np_test)
print(model.get_weights)
print(predictions)
print(predictions.shape)
while True:
    a, b = map(int, input().split())  #Это для того чтобы дальше посмотреть наглядно как работает модель, но она не работает)))
