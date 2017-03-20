import pickle
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
import numpy as np

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)



class DataSet(object):
    def __init__(self, datas, labels):
        self._datas = datas
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._num_examples = datas.shape[0]
    @property
    def datas(self):
        return self._datas

    @property
    def labels(self):
        return self._labels

    @property 
    def num_examples(self):
        return self._num_examples

    @property 
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._datas = self.datas[perm0]
            self._labels = self.labels[perm0]

        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start
            datas_rest_part = self._datas[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._datas = self.datas[perm]
                self._labels = self.labels[perm]
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            datas_new_part = self._datas[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((datas_rest_part.toarray(), datas_new_part.toarray()), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._datas[start:end].toarray(), self._labels[start:end]
def read_data_sets():
    train_datas, train_labes = create_datas(teachers_data_file='./data/teachers_data', courses_data_file='./data/courses_data')
    test_datas, test_labes = create_datas(teachers_data_file='./data/teachers_test_data', courses_data_file='./data/courses_test_data')
    return DataSet(datas=train_datas, labels=train_labes), test_datas, test_labes
def create_datas(teachers_data_file, courses_data_file):
    teachers_sparse_matrix = load_obj(teachers_data_file)
    courses_sparse_matrix = load_obj(courses_data_file)
    datas_matrix = vstack((teachers_sparse_matrix, courses_sparse_matrix))
    teachers_labels = np.append(np.ones(teachers_sparse_matrix.shape[0]), np.zeros(courses_sparse_matrix.shape[0]))
    course_labels = np.append(np.zeros(teachers_sparse_matrix.shape[0]), np.ones(courses_sparse_matrix.shape[0]))
    labels = np.vstack((teachers_labels, course_labels)).T
    return datas_matrix, labels

