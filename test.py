import torch
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from datasets import Dataset
# model_checkpoint = "microsoft/swin-tiny-patch4-window7-224" # pre-trained model from which to fine-tune
# batch_size = 32 # batch size for training and evaluation
#
# from datasets import load_dataset
# from datasets import load_metric
#
# metric = load_metric("accuracy")
#
# dataset = load_dataset("imagefolder", data_dir="../../2750")
#
# print(dataset)
# print(dataset['train'][20])

class TrainDataset(torch.utils.data.Dataset):
    """build training data-set"""
    def __init__(self, images, labels, transform=None, medical_transform=None):
        self.dict = []
        for i in range(len(images)):
            temp_dict = {'image': images[i], 'label': labels[i]}
            self.dict.append(temp_dict)
        self.images = images
        self.labels = labels
        self.transform = transform
        self.medical_transform = medical_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.dict[idx]

def build_dataset_camcan():
    """
    load Cam-Can MRI data
    https://www.cam-can.org
    """
    # load MRI data
    images, df = pickle.load(open('../../../mri_concat.pickle', 'rb'))
    # reformat data-frame
    df = df.reset_index()
    # retrieve the minimum, maximum and median age for skewed loss
    lim = (df['Age'].min(), df['Age'].max())
    median_age = df['Age'].median()
    # add color channel for images (bs, H, D, W) -> (bs, 1, H, D, W)
    images = np.expand_dims(images, axis=1)

    assert len(images.shape) == 5, images.shape
    assert images.shape[1] == 1
    assert len(images) == len(df)

    # assign a categorical label to Age for Stratified Split
    df['Age_categorical'] = pd.qcut(df['Age'], 25, labels=[i for i in range(25)])

    # Stratified train validation-test Split
    split = StratifiedShuffleSplit(test_size=0.2)
    train_index, validation_test_index = next(split.split(df, df['Age_categorical']))
    stratified_validation_test_set = df.loc[validation_test_index]
    assert sorted(train_index.tolist() + validation_test_index.tolist()) == list(range(len(df)))

    # Stratified validation test Split
    split2 = StratifiedShuffleSplit(test_size=0.5)
    validation_index, test_index = next(split2.split(stratified_validation_test_set,
                                                     stratified_validation_test_set['Age_categorical']))

    # NOTE: StratifiedShuffleSplit returns RangeIndex instead of the Original Index of the new DataFrame
    assert sorted(validation_index.tolist() + test_index.tolist()) == \
        list(range(len(stratified_validation_test_set.index)))
    assert sorted(validation_index.tolist() + test_index.tolist()) != \
        sorted(list(stratified_validation_test_set.index))

    # get the correct index of original DataFrame for validation/test set
    validation_index = validation_test_index[validation_index]
    test_index = validation_test_index[test_index]

    # ensure there is no duplicated index
    assert sorted(train_index.tolist() + validation_index.tolist() + test_index.tolist()) == list(range(len(df)))

    # get train/validation/test set
    train_images = images[train_index].astype(np.float32)
    validation_images = images[validation_index].astype(np.float32)
    test_images = images[test_index].astype(np.float32)
    # add dimension for labels: (32,) -> (32, 1)
    train_labels = np.expand_dims(df.loc[train_index, 'Age'].values, axis=1).astype(np.float32)
    validation_labels = np.expand_dims(df.loc[validation_index, 'Age'].values, axis=1).astype(np.float32)
    test_labels = np.expand_dims(df.loc[test_index, 'Age'].values, axis=1).astype(np.float32)

    # Pytorch Data-set for train set. Apply data augmentation if needed using "torchio"
    dataset_train = TrainDataset(images=train_images, labels=train_labels)

    return dataset_train

def function(dataset):
    ds = Dataset.from_generator(gen, gen_kwargs={"torch_dataset": dataset})
    return ds

def gen(torch_dataset):
    for idx in range(len(torch_dataset)):
        yield torch_dataset[idx]

if __name__ == '__main__':
    dataset = build_dataset_camcan()
    ds = function(dataset)
    print(ds)
    splits = ds.train_test_split(test_size=0.1)
    train_ds = splits["train"]
    val_ds = splits["test"]
    print(splits)
    print(train_ds)