import json
import pickle
from scipy.sparse import csr_matrix
import numpy as np

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def create_sparse_matrix_array(json_file_name, char_hash):
    with open(json_file_name) as data_file:
        datas = json.load(data_file)
    row = []
    col = []
    data = []
    for i in range(0, len(datas)):
        string = datas[i]
        for j in range(0, len(string)):
            row.append(i)
            col.append(char_hash[string[j]])
            data.append(1)
    matrix = csr_matrix((np.array(data), (np.array(row), np.array(col))), shape=(len(datas), len(char_hash)))
    return matrix

char_hash = load_obj('./data/char_hash')
matrix = create_sparse_matrix_array('./original_json_files/courses_src.json', char_hash)
split_index = int(matrix.shape[0] * 0.9)
save_obj(matrix[0:split_index], "./data/courses_data")
save_obj(matrix[split_index:-1], './data/courses_test_data')
matrix = create_sparse_matrix_array('./original_json_files/teachers_src.json', char_hash)
split_index = int(matrix.shape[0] * 0.9)
save_obj(matrix[0:split_index], "./data/teachers_data")
save_obj(matrix[split_index:-1], './data/teachers_test_data')
