from Transapi import TransNASBenchAPI as API
import numpy as np
import pickle
import copy
from nasbench import api as nas101api

ope_list = ['None', 'Skip-connection', '1*1_conv', '3*3_conv']
LABEL2ID = {
    "input": -1,
    "output": -2,
    "1*1_conv": 0,
    "3*3_conv": 1
}
BASIC_MATRIX = [[0, 1, 1, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0, 0]]
def OOE_micro2matrix(micro_encode):
    
    """
    input: An encode of the transNAS dataset. For example, "1_02_333".
    output: A matrix of a micro-model.
    """
    
    OOE_adjacency_matrix = np.zeros((4, 4))
    cell_config = micro_encode.split("_")
    for i in range(len(cell_config)):
        for k in range(len(cell_config[i])):
            OOE_adjacency_matrix[k, i + 1] = int(cell_config[i][k])
    return OOE_adjacency_matrix
def cell_encode_trans(micro_model):
    """
    Obtain the cell encode with transNAS
    input: micro_model
    output: cell_encode 
    """
    cell_encode = micro_model.split("-")[2]
    return cell_encode
def get_task_performance(api, micro_model):
    """
    Obtain the model performance of different architecture
    input: api of TransNAS, micro_model
    output: micor_info
    """
    micor_info  = {}
    for xtask in api.task_list:
        # print(xtask)
        if xtask in ['class_scene', 'class_object', 'jigsaw']:
            xmetric = 'test_top1'
            accuracy = api.get_single_metric(micro_model, xtask, xmetric, mode='best')
            # print(accuracy)
            micor_info[xtask] = accuracy
        elif xtask in ['room_layout', 'segmentsemantic']:
            xmetric = 'test_loss'
            test_loss = api.get_best_epoch_status(micro_model, xtask, metric=xmetric)['test_loss']
            micor_info[xtask] = test_loss
            # print(neg_loss)
        else:
            xmetric = 'test_ssim'
            test_ssim = api.get_best_epoch_status(micro_model, xtask, metric=xmetric)['test_ssim']
            # print(neg_loss)
            micor_info[xtask] = test_ssim
    for xinfo in api.info_names:
        xinfo_perf = api.get_model_info(micro_model, xtask, xinfo)
        micor_info[xinfo] = xinfo_perf
        # print(xinfo_perf)
    return micor_info
def cell_encode2opelist(cell_encode):
    cell_string = cell_encode.replace("_", "")
    # print(cell_string)
    op_list = []
    for cs in cell_string:
        op_list.append(ope_list[int(cs)])
    return op_list

def OON_ops2matrixandopu(ops):
        # delete the skip connections nodes and the none nodes
        # output the pruned metrics
        # start to change matrix
        matrix = copy.deepcopy(BASIC_MATRIX)
        for i, op in enumerate(ops, start=1):
            m = []
            n = []
            if op == 'Skip-connection':
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

            elif op == 'None':
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
def ops_list2int(ops_list):
    module_integers = np.array([LABEL2ID[x] for x in ops_list])
    return module_integers
def ops2onehot(opsint):
    ops_onehot = np.array([[i == k + 2 for i in range(4)] for k in opsint], dtype=np.float32)
    return ops_onehot
def main():
    path2nas_bench_file = "./Datas/transnas-bench_v10141024.pth"
    api = API(path2nas_bench_file)
    all_arch_dict = api.all_arch_dict  # {search_space : list_of_architecture_names}
    micro_spaces = all_arch_dict["micro"]
    all_micro = {}
    for idx, micro_model in enumerate(micro_spaces):
        all_info_micro = {}
        micro_info = get_task_performance(api=api, micro_model = micro_model)
        cell_encoding = cell_encode_trans(micro_model = micro_model)
        OOE_matrix = OOE_micro2matrix(cell_encoding)
        ops = cell_encode2opelist(cell_encode=cell_encoding)
        OON_matrix, ops_without_unless = OON_ops2matrixandopu(ops)
        if ops_without_unless==None:
            continue
        else:
            
            opsint = ops_list2int(ops_without_unless)
            opsone_hot = ops2onehot(opsint)
            OON_matrix, opsone_hot = padding_OON(OON_matrix,opsone_hot)
            all_info_micro['micro_info'] = micro_info
            all_info_micro['cell_encoding'] = cell_encoding
            all_info_micro['OOE_matrix'] = OOE_matrix
            all_info_micro['ops'] = ops
            all_info_micro['adjacency'] = OON_matrix
            all_info_micro['ops_without_unless'] = ops_without_unless
            all_info_micro['opsint'] = opsint
            all_info_micro['operations'] = opsone_hot
            # print(all_info_micro)
        all_micro[micro_model] = all_info_micro
    # print(len(all_micro))
    with open("./Datas/all_trans101_micro.pickle",'wb') as file:
        pickle.dump(all_micro, file)
def get_mean_std():
    with open("./Datas/all_micro.pickle", 'rb') as file:
            sample = pickle.load(file)
            accs = []
            
            for arch in sample.keys():
                # all_arch.append(arch)
                accs.append(sample[arch]["micro_info"]['class_object'])
                print(accs)
            print(np.mean(accs))
            print(np.std(accs))
if __name__ == '__main__':
    main()