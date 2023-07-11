from transformers import AutoImageProcessor
from torchvision.transforms import CenterCrop, Compose, Normalize, RandomHorizontalFlip, \
    RandomResizedCrop, Resize, ToTensor

import logging
import numpy as np
from PIL import Image
import config

logger = logging.getLogger(__name__)

# Apply AutoImageProcessor to preprocess images.
# It should be noted that HF dataset built from ImageFolder method stores images as PIL object. 
# If the dataset is built from a PT dataset, the images would be in array format.


def preprocess_datasets(dataset_train, dataset_val, dataset_test, args):
    """Main function to preprocess the dataset"""
    # initialize transformation for train/val/test set
    config.train_transforms, config.val_transforms, config.test_transforms, image_processor = preprocess(args)
    # transform the dataset
    if args.dataset_build_method == 'imagefolder':
        dataset_train.set_transform(preprocess_train)
        dataset_val.set_transform(preprocess_val)
        dataset_test.set_transform(preprocess_test)
    elif args.dataset_build_method == 'PT2HF':
        dataset_train.set_transform(preprocess_train_pt)
        dataset_val.set_transform(preprocess_val_pt)
        dataset_test.set_transform(preprocess_test_pt)

    # set a checker to check transformed data
    idx = len(dataset_train) // 2
    logger.info(f"The type of model input is {type(dataset_train[idx]['pixel_values'])}, "
                f"output is {type(dataset_train[idx]['label'])}. "
                f"The shape of model input is {dataset_train[idx]['pixel_values'].shape}")

    return dataset_train, dataset_val, dataset_test, image_processor


def preprocess(args):
    """Define transformations for train/val/test set"""
    image_processor = AutoImageProcessor.from_pretrained(args.model_checkpoint)
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

    # set transform
    train_transforms = Compose([RandomResizedCrop(image_processor.size["height"]),
                                RandomHorizontalFlip(),
                                ToTensor(),
                                normalize])

    val_transforms = Compose([Resize(image_processor.size["height"]),
                              CenterCrop(image_processor.size["height"]),
                              ToTensor(),
                              normalize])

    test_transforms = Compose([Resize(image_processor.size["height"]),
                               ToTensor(),
                               normalize])

    return train_transforms, val_transforms, test_transforms, image_processor


def preprocess_train_pt(example_batch):
    """
    Apply train_transforms across a batch.
    Note that set_transform will replace the format defined by set_format and
        will be applied on-the-fly (when it is called).
    In build_dataset.py, we use set_format method to set both image and label to torch.tensor, while
    it will be rewritten when we call the set_transform method.
    """
    # transform data type to PIL image for later convenience
    example_batch["pixel_values"] = [config.train_transforms(Image.fromarray(np.uint8(np.array(image)))) for image in example_batch["image"]]
    return example_batch

def preprocess_val_pt(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [config.val_transforms(Image.fromarray(np.uint8(np.array(image)))) for image in example_batch["image"]]
    return example_batch

def preprocess_test_pt(example_batch):
    """Apply test_transforms across a batch."""
    example_batch["pixel_values"] = [config.test_transforms(Image.fromarray(np.uint8(np.array(image)))) for image in example_batch["image"]]
    return example_batch


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [config.train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [config.val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

def preprocess_test(example_batch):
    """Apply test_transforms across a batch."""
    example_batch["pixel_values"] = [config.test_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch
