import argparse
import logging
import os
import torch

from utils.build_dataset import build_datasets
from utils.build_processor import preprocess_datasets
from utils.build_models import build_model
from utils.build_training import build_train_argument, build_trainer, train_val_test
from utils.build_training_loop import build_data_loader, build_optimizer, train_val_test_pt
from utils.utils import RunManager, update_args

import config
config.init()

# create logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
# create results folder
results_folder = 'model_ckpt_results'
os.makedirs(results_folder, exist_ok=True)
os.makedirs(results_folder + '/log', exist_ok=True)


def build_parser():
    """
    build parser for classic image classification.
    A template for running the code through the terminal is listed below:
    python main.py --comment run0
    """
    parser = argparse.ArgumentParser(description='Image Classification')
    # model settings
    parser.add_argument('--model', type=str, default='vit',
                        choices=['vit'], help='model configurations')
    parser.add_argument('--model-checkpoint', type=str, default='google/vit-base-patch16-224-in21k',
                        help='pretrained model checkpoint')
    parser.add_argument('--peft', action='store_true', default=True,
                        help='use peft to perform fine-tuning')
    parser.add_argument('--peft-config', type=str, default='lora',
                        choices=['lora'], help='peft configurations')
    # dataset settings
    parser.add_argument('--dataset-dir', type=str, default='ImageFolder',
                        help='path to dataset directory')
    parser.add_argument('--dataset-build-method', type=str, default='imagefolder', choices=['imagefolder', 'PT2HF'], 
                        help='''methods to build the dataset. 
                                "imagefolder": HF ImageFolder method; 
                                "PT2HF": Pytorch dataset to HF dataset''')
    parser.add_argument('--val-test-size', type=float, default=0.5,
                        help='proportion of validation & test set of the total dataset')
    parser.add_argument('--test-size', type=float, default=0.5,
                        help='proportion of test set of the "validation & test" set')
    parser.add_argument('--random-state', type=int, default=1000,
                        help='used in train/val/test split')
    # training settings
    parser.add_argument('--use-trainer', action='store_true', default=False,
                        help='whether to use HF trainer to train models')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--num-train-epochs', type=int, default=2, help='number of epoch')
    # optimizer settings
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--lr-scheduler-type', type=str, default='linear', help='lr scheduler type')
    parser.add_argument('--num-warmup-steps', type=int, default=0, help='num warmup steps')
    # testing settings
    parser.add_argument('--test', action='store_true', default=False,
                        help='test trained model performance')
    # default settings
    parser.add_argument('--comment', type=str, default='run0',
                        help='comments to distinguish different runs')
    return parser


def main():
    """overall workflow for Image Classification"""
    # build parser
    args = build_parser().parse_args()
    logger.info(f'Parser arguments are {args}')

    # use CUDA if possible
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.cuda = True if device == 'cuda' else False
    logger.info(f'Found device: {device}')

    # update args
    args = update_args(args=args)

    # build dataset and dataloader
    # label2id and id2label mapping are used to change the number of output class for pretrained models 
    dataset_train, dataset_val, dataset_test, label2id, id2label = build_datasets(args)
    logger.info('Dataset loaded')

    # build image processor 
    dataset_train, dataset_val, dataset_test, image_processor = preprocess_datasets(dataset_train, dataset_val, dataset_test, args)
    logger.info('Dataset preprocessed')

    # build models
    model = build_model(args=args, label2id=label2id, id2label=id2label)
    logger.info('Model loaded')

    # train and evaluate models
    if args.use_trainer:
        # Trainer model will be saved in "model_ckpt_results/model_name"
        # build HF trainer
        train_args = build_train_argument(args=args)
        trainer = build_trainer(model=model, train_args=train_args, dataset_train=dataset_train, 
                                dataset_val=dataset_val, image_processor=image_processor)
        logger.info('Trainer initialized')

        # Train and evaluate!
        train_val_test(trainer=trainer, dataset_test=dataset_test, args=args)

    else: 
        # Use native Pytorch loop. 
        # NO Trainer model will be saved in "model_ckpt_results/model_name-pt"
        os.makedirs(args.out_dir_no_trainer, exist_ok=True)

        # build dataloader
        train_loader, val_loader, test_loader = build_data_loader(
            args=args, dataset_train=dataset_train, dataset_val=dataset_val, dataset_test=dataset_test)
            
        # build dataloader
        optimizer, lr_scheduler = build_optimizer(model=model, train_loader=train_loader, args=args)

        # build RunManager to save stats from training
        m = RunManager(args=args)
        m.begin_run(train_loader, val_loader, test_loader)

        # train and evaluate
        model.to(device)
        train_val_test_pt(args=args, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, 
            model=model, device=device, optimizer=optimizer, lr_scheduler=lr_scheduler, m=m, 
            label2id=label2id, id2label=id2label)

    logger.info('Model finished!')


if __name__ == '__main__':
    main()
