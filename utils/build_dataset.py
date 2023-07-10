from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset
from utils.utils import TrainDataset, ValidationDataset, TestDataset

import numpy as np
import os, glob, cv2

# Task: Given a folder with images, load images as an HF dataset
# Solution: 1. Use HF ImageFolder method.
#           2. Load the images as a Pytorch dataset and transform into an HF dataset.
#
# https://huggingface.co/docs/datasets/create_dataset


def build_datasets(args):
    """Whole function for building the dataet"""
    if args.dataset_build_method == 'imagefolder':
        dataset_train, dataset_val, dataset_test = build_dataset_from_image_folder(args=args)
        label2id, id2label = build_id_label_mapping(dataset=dataset_train)
    elif args.dataset_build_method == 'PT2HF':
        images, labels, id2label, label2id = load_image_from_directory(args=args)
        dataset_train, dataset_val, dataset_test = build_dataset_from_array(images, labels, args)
    
    return dataset_train, dataset_val, dataset_test, label2id, id2label


def build_dataset_from_image_folder(args):
    """HF dataset could be built from a folder with specific structures"""
    dataset = load_dataset("imagefolder", data_dir=f"{args.dataset_dir}")['train']
    # Split into train/val/test set (90/5/5)
    train_valtest = dataset.train_test_split(test_size=args.val_test_size, stratify_by_column="label")
    test_valid = train_valtest['test'].train_test_split(test_size=args.test_size, stratify_by_column="label")

    # Define train/val/test sets
    dataset_train = train_valtest['train']
    dataset_val = test_valid['train']
    dataset_test = test_valid['test']

    return dataset_train, dataset_val, dataset_test


def build_id_label_mapping(dataset):
    labels = dataset.features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    return label2id, id2label


def load_image_from_directory(args):
    """Load images and labels from a given directory"""
    # Get all classes
    classes = list()
    for subfolder_name in os.listdir(args.dataset_dir):
        if not subfolder_name.startswith('.'):
            classes.append(subfolder_name)

    # Create id2label and label2id dict
    label2id, id2label = dict(), dict()
    for i, label in enumerate(classes):
        label2id[label] = i
        id2label[i] = label

    # load images and labels into array
    image_list, label_list = list(), list()
    files = glob.glob(args.dataset_dir + '/**/*.jpg', recursive=True)
    for file in files:
        image_class = label2id[os.path.basename(file).split('_')[0]]
        image = np.array(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB))

        image_list.append(image)
        label_list.append(image_class)
    
    images = np.asarray(image_list)
    labels = np.array(label_list).reshape(-1, 1)

    return images, labels, id2label, label2id


def build_dataset_from_array(images, labels, args):
    """Build PT dataset from array and transform to HF dataset"""
    # stratified train/val/test split
    x_train, x_valtest, y_train, y_valtest = train_test_split(images, labels, test_size=args.val_test_size, stratify=labels)
    x_val, x_test, y_val, y_test = train_test_split(x_valtest, y_valtest, test_size=args.test_size, stratify=y_valtest)

    # build PT dataset
    dataset_train_pt = TrainDataset(images=x_train, labels=y_train, huggingface=True)
    dataset_validation_pt = ValidationDataset(images=x_val, labels=y_val, huggingface=True)
    dataset_test_pt = TestDataset(images=x_test, labels=y_test, huggingface=True)

    # transform PT dataset to HF dataset
    dataset_train = transform_to_huggingface_dataset(pt_dataset=dataset_train_pt)
    dataset_train.set_format(type='torch', columns=['image', 'label'])
    dataset_validation = transform_to_huggingface_dataset(pt_dataset=dataset_validation_pt)
    dataset_validation.set_format(type='torch', columns=['image', 'label'])
    dataset_test = transform_to_huggingface_dataset(pt_dataset=dataset_test_pt)
    dataset_test.set_format(type='torch', columns=['image', 'label'])

    return dataset_train, dataset_validation, dataset_test


def transform_to_huggingface_dataset(pt_dataset):
    """transform a pytorch dataset into a huggingface dataset"""
    hf_dataset = Dataset.from_generator(generator, gen_kwargs={"pt_dataset": pt_dataset})
    return hf_dataset


def generator(pt_dataset):
    """generator function"""
    for idx in range(len(pt_dataset)):
        yield pt_dataset[idx]
