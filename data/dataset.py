from torch.utils.data.dataloader import  Dataset 




class CombinedDataset(Dataset):
    def __init__(self, sensor1, sensor2):
        self.dataset1 = sensor1
        self.dataset2 = sensor2

    def __getitem__(self, i):
        return self.dataset1[i],self.dataset2[i]

    def __len__(self):
        return len(self.dataset1)