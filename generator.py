# System
import keras
import numpy as np
from multiprocessing import Pool

def load_data(path):
    global cacher
    embedding = np.load(path, allow_pickle=True)
    # embedding = np.expand_dims(embedding, -1)
    return embedding


def load_Couple(path):
    global cacher
    embedding_1 = np.load(path[0], allow_pickle=True)
    embedding_2 = np.load(path[1], allow_pickle=True)
    # embedding = np.expand_dims(embedding, -1)
    return embedding_1, embedding_2


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_IDs, labels, augmentation=True, batch_size=32,
                 n_classes=2, shuffle=True, mp_pooler=12):
        """Initialization"""
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.batch_size = 2*(batch_size//2)
        self.augmentation = augmentation
        self.mp_pooler = Pool(mp_pooler)
        self.identity_dict = dict()
        for (label, utterance) in zip(labels, list_IDs):
            if self.identity_dict.get(label, False):
                self.identity_dict[label] += [utterance]
            else:
                self.identity_dict[label] = [utterance]
        self.identities = list(self.identity_dict.keys())
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.identities) / self.batch_size))*100

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if index > int(np.floor(len(self.identities) / self.batch_size)):
            index %= int(np.floor(len(self.identities) / self.batch_size))
            if index == 0:
                np.random.shuffle(self.identities)
        key_identities = self.identities[index*self.batch_size//2:(index+1)*self.batch_size//2]
        adversarial_identities = self.identities[(index+1)*self.batch_size//2:(index+2)*self.batch_size//2]

        key_utterances = np.array([np.random.choice(self.identity_dict[i], 2, replace=False) for i in key_identities])
        adversarial_utterances = np.array([np.random.choice(self.identity_dict[i], 1) for i in adversarial_identities])

        ref_utterances = key_utterances[:, 0]
        true_utterances = key_utterances[:, 1]
        false_utterances = adversarial_utterances[:, 0]

        ones = np.ones(self.batch_size // 2)
        zeros = np.zeros([self.batch_size//2])
        y = np.concatenate([ones, zeros])
        # y = keras.utils.to_categorical(y, num_classes=self.n_classes)
        # Generate data
        X = self.__data_generation_mp(ref_utterances, true_utterances, false_utterances)
        return X, [y]


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.identities = np.array(list(self.identity_dict.keys()))
        if self.shuffle:
            np.random.shuffle(self.identities)



    def __data_generation_mp(self, ref_utterances, true_utterances, false_utterances):
        X_1 = self.mp_pooler.map(load_data, np.concatenate([ref_utterances, ref_utterances]))
        X_2 = self.mp_pooler.map(load_data, np.concatenate([true_utterances, false_utterances]))

        # X = np.expand_dims(np.array([p.get() for p in X]), -1)
        return np.array(list(zip(X_1, X_2)))

class TestGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, list_couples, labels, batch_size=32, n_classes=2, mp_pooler=12):
        """Initialization"""
        self.n_classes = n_classes
        self.list_couples = list_couples
        self.labels = labels
        self.batch_size = 2*(batch_size//2)
        self.mp_pooler = Pool(mp_pooler)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(len(self.list_couples) // self.batch_size)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        coupels = self.list_couples[index*self.batch_size:(index+1)*self.batch_size]

        labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation_mp(couples=coupels, labels=labels)
        return X, y

    def on_epoch_end(self):
        pass

    def __data_generation_mp(self, couples, labels):
        X = self.mp_pooler.map(load_Couple, couples)
        # X = np.expand_dims(np.array([p.get() for p in X]), -1)
        y = labels
        return np.array(X), y #keras.utils.to_categorical(y, num_classes=self.n_classes)

def OHEM_generator(model, datagen, steps, propose_time, batch_size, dims, nclass):
    # propose_time : number of candidate batches.
    # prop : the number of hard batches for training.
    step = 0
    interval = np.array([i*(batch_size // propose_time) for i in range(propose_time)] + [batch_size])

    while True:
        if step == 0 or step > steps - propose_time:
            step = 0
            datagen.on_epoch_end()

        # propose samples,
        samples = np.empty((batch_size,) + dims)
        targets = np.empty((batch_size, nclass))

        for i in range(propose_time):
            x_data, y_data = datagen.__getitem__(index=step+i)
            preds = model.predict(x_data, batch_size=batch_size)   # prediction score
            errs = np.sum(y_data * preds, -1)
            err_sort = np.argsort(errs)

            indices = err_sort[:(interval[i+1]-interval[i])]
            samples[interval[i]:interval[i+1]] = x_data[indices]
            targets[interval[i]:interval[i+1]] = y_data[indices]

        step += propose_time
        yield samples, targets
