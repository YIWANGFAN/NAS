import pickle
import os
import collections
from nas_201_api import NASBench201API as API201
import copy
from nasbench import api as nas101api
import numpy as np
LABEL2ID = {
    "input": -1,
    "output": -2,
    "nor_conv_1x1": 0,
    "nor_conv_3x3": 1,
    "avg_pool_3x3": 2
}
BASIC_MATRIX = [[0, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0]]


def save_arch_str2op_list(save_arch_str):
        op_list = []
        save_arch_str_list = API201.str2lists(save_arch_str)
        op_list.append(save_arch_str_list[0][0][0])
        op_list.append(save_arch_str_list[1][0][0])
        op_list.append(save_arch_str_list[1][1][0])
        op_list.append(save_arch_str_list[2][0][0])
        op_list.append(save_arch_str_list[2][1][0])
        op_list.append(save_arch_str_list[2][2][0])
        return op_list
def OON_ops2matrixandopu(ops):
        # delete the skip connections nodes and the none nodes
        # output the pruned metrics
        # start to change matrix
        matrix = copy.deepcopy(BASIC_MATRIX)
        for i, op in enumerate(ops, start=1):
            m = []
            n = []
            if op == 'skip_connect':
                for m_index in range(8):
                    ele = matrix[m_index][i]
                    if ele == 1:
                        # set element to 0
                        matrix[m_index][i] = 0
                        m.append(m_index)

                for n_index in range(8):
                    ele = matrix[i][n_index]
                    if ele == 1:
                        # set element to 0
                        matrix[i][n_index] = 0
                        n.append(n_index)

                for m_index in m:
                    for n_index in n:
                        matrix[m_index][n_index] = 1

            elif op == 'none':
                for m_index in range(8):
                    matrix[m_index][i] = 0
                for n_index in range(8):
                    matrix[i][n_index] = 0
        ops_copy = copy.deepcopy(ops)
        ops_copy.insert(0, 'input')
        ops_copy.append('output')

        # start pruning
        model_spec = nas101api.ModelSpec(matrix=matrix, ops=ops_copy)
        # print(model_spec.matrix, model_spec.ops)
        return model_spec.matrix, model_spec.ops
def ops_list2int(ops_list):
    print(ops_list)
    module_integers = np.array([LABEL2ID[x] for x in ops_list])
    return module_integers
def ops2onehot(opsint):
    ops_onehot = np.array([[i == k + 2 for i in range(4)] for k in opsint], dtype=np.float32)
    return ops_onehot
def padding_OON(matrix,ops):
    n = len(matrix)
    # print(n)
    if n != 8:
        for i in range(0,8-n):
            matrix = np.pad(matrix, ((0, 1), (0, 0)), mode='constant')
            matrix = np.pad(matrix, ((0, 0), (0, 1)), mode='constant')
            ops = np.pad(ops, ((0, 1), (0, 0)), mode='constant')
    ops = np.pad(ops,((0, 0), (0, 1)), mode='constant')
    assert len(ops) == len(matrix)
    assert len(matrix) == 8
    assert len(matrix[:,0]) == 8
    assert len(ops[0]) == 5
    return matrix, ops
def main():
    nasbench201 = API201(r'./Datas/NAS-Bench-201-v1_1-096897.pth')
    all_micro = {}
    for index in range(len(nasbench201.evaluated_indexes)):
        all_info_micro = {}
        info = nasbench201.query_meta_info_by_index(index, '12')
        arch_str = info.arch_str
        op_list = save_arch_str2op_list(arch_str)
        cifar10_valid = info.get_metrics('cifar10-valid', 'x-valid')['accuracy']
        print(cifar10_valid)
        # num_vertices = len(op_list)
        NAS201_matrix, NAS201_ops_list = OON_ops2matrixandopu(ops=op_list)
        # print(NAS201_matrix, NAS201_ops_list)
        if NAS201_ops_list==None:
                # print(micro_model, micro_info)
                continue
        else:
            opsint = ops_list2int(NAS201_ops_list)
            opsone_hot = ops2onehot(opsint)
            NAS201_matrix, opsone_hot = padding_OON(NAS201_matrix,opsone_hot)
            all_info_micro['ops'] = op_list
            all_info_micro['accuracy_cifar10'] = cifar10_valid
            all_info_micro['adjacency'] = NAS201_matrix
            all_info_micro['ops_without_unless'] = NAS201_ops_list
            all_info_micro['opsint'] = opsint
            all_info_micro['operations'] = opsone_hot
        all_micro[arch_str] = all_info_micro
    with open("./Datas/all_NAS201_padding_micro.pickle",'wb') as file:
        pickle.dump(all_micro, file)
def get_mean_std():
    with open("./Datas/all_NAS201_padding_micro.pickle", 'rb') as file:
        sample = pickle.load(file)
        accs = []
        for arch in sample.keys():
            # all_arch.append(arch)
            accs.append(sample[arch]["accuracy_cifar10"])
            # print(accs)
        print(np.mean(accs))
        print(np.std(accs))
if __name__ == '__main__':
    main()
    