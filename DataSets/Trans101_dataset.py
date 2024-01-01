from torch.utils.data import Dataset
import pickle
from torch.utils.data import DataLoader

class Trans101Dataset(Dataset):
    MEAN = 46.94359937793643
    STD = 5.903938447889667
    def __init__(self, data_path) -> None:
        with open(data_path, 'rb') as file:
            self.sample = pickle.load(file)
        self.all_arch = []
        for arch in self.sample.keys():
            self.all_arch.append(arch)
    def __len__(self):
        return len(self.sample)
    @classmethod
    def normalize(cls, num):
        return (num - cls.MEAN) / cls.STD

    @classmethod
    def denormalize(cls, num):
        return num * cls.STD + cls.MEAN
    
    
    def __getitem__(self, index):
        arch = self.all_arch[index]
        arch_class_object = self.sample[arch]["micro_info"]['class_object']
        arch_OON_matrix = self.sample[arch]['adjacency']
        arch_ops_one_hot = self.sample[arch]['operations']
        num_vertices = len(self.sample[arch]['adjacency'])
        
        result = {
            'arch_class_object': self.normalize(arch_class_object),
            'adjacency': arch_OON_matrix,
            'operations': arch_ops_one_hot,
            'num_vertices': num_vertices
        }
        
        return result
    
# if __name__ == '__main__':
#     data = Trans101Dataset("./api_home/padding_all_micro.pickle")
#     # print(data)
#     data_loader = DataLoader(data, batch_size=32, shuffle=True, drop_last=True)
#     for step, batch in enumerate(data_loader):
#         print(batch)