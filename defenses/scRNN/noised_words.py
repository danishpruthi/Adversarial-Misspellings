import random

ALPHABAET_LENGTH = 26


def noise_string(string, noise_amount):
    '''


    :param string:
    :param noise_amount:
    :return: all strings *UP* to the noise_amount
    '''
    if noise_amount == 0:
        return string
    else:
        word = noise_step(string)
        return noise_string(word, noise_amount - 1)


def noise_step(str):
    if len(str) <=1:
        return str
    operation = random.randint(0, 3)
    if operation == 0:
        # insertion
        index = random.randint(0, len(str))
        char = random.randrange(0, ALPHABAET_LENGTH)
        return str[:index] + chr(ord('a') + char) + str[index:]
    elif operation == 1:
        # deletion
        index = random.randrange(0, len(str))
        return str[:index] + str[index + 1:]
    elif operation == 2:
        # substitution
        index = random.randrange(0, len(str))
        char = random.randrange(0, ALPHABAET_LENGTH)
        return str[:index] + chr(ord('a') + char) + str[index + 1:]
    elif operation == 3:
        # transposition
        index = random.randrange(0, len(str) - 1)
        return str[:index] + str[index + 1] + str[index] + str[index + 2:]


def rnd_noise_sentence(sentence1, sentence2, sentence1_key=None, sentence2_key=None, rate=0.0):
    rnd_sent = []
    for word in sentence1.split():
        if random.random() < rate:
            noised_list = list(noise_string(word, 1))
            rnd_str = noised_list[random.randint(0, len(noised_list) - 1)]
            rnd_sent.append(rnd_str)
        else:
            rnd_sent.append(word)
    noised_sent1 = " ".join([i for i in rnd_sent])
    if sentence2_key is None:
        return {sentence1_key: noised_sent1}
    rnd_sent = []
    for word in sentence2.split():
        if random.random() < rate:
            noised_list = list(noise_string(word, 1))
            rnd_str = noised_list[random.randint(0, len(noised_list) - 1)]
            rnd_sent.append(rnd_str)
        else:
            rnd_sent.append(word)
    noised_sent2 = " ".join([i for i in rnd_sent])
    return {sentence1_key: noised_sent1, sentence2_key: noised_sent2}


def rnd_noise_sentence_word(sentence1, sentence2, sentence1_key=None, sentence2_key=None, rate=0.0,noise_amount=1):
    rnd_sent = []
    for word in sentence1.split():
        if random.random() < rate:
            rnd_str = noise_string(word, noise_amount)
            rnd_sent.append(rnd_str)
        else:
            rnd_sent.append(word)
    noised_sent1 = " ".join([i for i in rnd_sent])
    if sentence2_key is None:
        return {sentence1_key: noised_sent1}
    rnd_sent = []
    for word in sentence2.split():
        if random.random() < rate:
            rnd_str = noise_string(word, noise_amount)
            rnd_sent.append(rnd_str)
        else:
            rnd_sent.append(word)
    noised_sent2 = " ".join([i for i in rnd_sent])
    return {sentence1_key: noised_sent1, sentence2_key: noised_sent2}
# for i in range(10):
#     print(noise_string('you better run you better say hello!', 3))
