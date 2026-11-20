import sqlite3
from torch.utils.data import DataLoader,Dataset
import torch
import pandas as pd
import random
import numpy as np
from torch.utils.data.sampler import Sampler
DB_PATH = "../METADDIEdata/event_META_DDIE_5foldLabel.db"
# DB_PATH = "../METADDIEdata/drugbank_few_shot_DDIE.db"
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

class MetaDDIETask(object):
    def __init__(self,all_event,num_ways,support_num,query_num,state,excludeLabel):
        self.num_ways = num_ways
        self.support_num = support_num
        self.query_num = query_num
        # common 事件：训练集和验证集都从 1～all_event 中采样事件
        if state == "train" or state == "val":
            select_class = random.sample(range(1, all_event + 1), num_ways)

        # fewer 测试事件：176～204  63～75
        elif state == 'test' or state == 'fewer':
            select_class = random.sample(range(all_event + 1, 205), num_ways)  # 203  191   205
            # select_class = random.sample(range(all_event + 1, 76), num_ways)

        # rare 测试事件：205～227   76～86
        elif state == 'rare':
            select_class = random.sample(range(205, 228), num_ways)
            # select_class = random.sample(range(76, 87), num_ways)

        else:
            raise ValueError(
                f"Unknown state: {state}. "
                f"Expected train, val, test, fewer or rare."
            )
        labels = np.array(range(num_ways))
        labels = dict(zip(select_class,labels))
        samples = {}
        self.support_samples = []
        self.query_samples = []

        for c in select_class:
            temp = cur.execute( 'select * from event' + str(c)).fetchall()

            # common 训练集：
            # excludeLabel=4 时，使用 fold 0～3
            if state == 'train':
                temp = [x for x in temp if int(x[4]) != int(excludeLabel)]   # meta
                # temp = [x for x in temp if int(x[5]) != int(excludeLabel)]   # drugbank

            # common 验证集：
            # excludeLabel=4 时，只使用 fold 4
            if state == 'val':
                temp = [x for x in temp if int(x[4]) == int(excludeLabel)]   # meta
                # temp = [x for x in temp if int(x[5]) == int(excludeLabel)]   # drugbank

            # state == 'test' 时不按照 common 的 fold 筛选
            # fewer：176～204
            # rare：205～227

            samples[c] = random.sample(temp, support_num + query_num)
            self.support_samples += samples[c][:support_num]
            self.query_samples += samples[c][support_num:support_num + query_num]

        self.support_labels = [labels[int(x[3])] for x in self.support_samples]    # meta
        self.query_labels = [labels[int(x[3])] for x in self.query_samples]     # meta
        # self.support_labels = [labels[int(x[4])] for x in self.support_samples]   # drugbank
        # self.query_labels = [labels[int(x[4])] for x in self.query_samples]   # drugbank
class FewShotDataset(Dataset):
    def __init__(self, task, split='train', transform=None, target_transform=None):
        #self.df_event = pd.read_sql('select * from newEvent;', conn)
        #self.transform = transform # Torch operations on the input image
        #self.target_transform = target_transform
        self.task = task
        self.split = split
        self.data = self.task.support_samples if self.split == 'train' else self.task.query_samples
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class DDIEDataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(DDIEDataset, self).__init__(*args,**kwargs)

    def __getitem__(self, idx):
        DDIE = self.data[idx]
        drug1 = DDIE[1]
        drug2 = DDIE[2]
        label = self.labels[idx]
        return drug1,drug2,label

class ClassBalancedSampler(Sampler):
    def __init__(self,num_per_class,num_ways,num_instances,shuffle=True):
        self.num_per_class=num_per_class
        self.num_ways = num_ways
        self.num_instances = num_instances
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            batch = [[i+j*self.num_instances for i in torch.randperm(self.num_instances)[:self.num_per_class]] for j in range(self.num_ways)]
        else:
            batch = [[i+j*self.num_instances for i in range(self.num_instances)[:self.num_per_class]] for j in range(self.num_ways)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task,num_per_class=1,split='train',shuffle=True):
    dataset = DDIEDataset(task,split=split)

    if split=='train':
        sampler = ClassBalancedSampler(num_per_class,task.num_ways,task.support_num,shuffle=shuffle)
    else:
        sampler = ClassBalancedSampler(num_per_class,task.num_ways,task.query_num,shuffle=shuffle)
    loader=DataLoader(dataset,batch_size=num_per_class*task.num_ways,sampler=sampler)

    return loader