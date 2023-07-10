from collections import OrderedDict
import torch
import time, os, json
import logging
import pandas as pd

logger = logging.getLogger(__name__)
results_folder = 'model_ckpt_results'

# ------------------- update arguments ---------------------

def update_args(args):
    args.out_dir = results_folder
    args.log_dir = results_folder + '/log'

    args.model_name = args.model_checkpoint.split("/")[-1]
    args.best_model_ckpt_dir = os.path.join(args.out_dir, args.model_name, 'trainer_state.json')
    # used in "no_trainer"
    args.model_config = args.model_checkpoint.split("/")[-1] + '-pt'
    args.out_dir_no_trainer = f'{args.out_dir}/{args.model_config}'
    return args

# ------------------- Pytorch Dataset ---------------------

class TrainDataset(torch.utils.data.Dataset):
    """
    build training dataset
    Note that Huggingface requires that __getitem__ method returns dict object
    """
    def __init__(self, images, labels, transform=None, huggingface=True):
        self.huggingface = huggingface

        if self.huggingface:
            self.dict = []
            for i in range(len(images)):
                temp_dict = {'image': images[i], 'label': labels[i]}
                self.dict.append(temp_dict)
        else:
            self.images = images
            self.labels = labels
            self.transform = transform

    def __len__(self):
        if self.huggingface:
            return len(self.dict)
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.huggingface:
            return self.dict[idx]
        else:
            # In ordinary Pytorch dataset, augmentation is performed within Dataset Class
            image, label = self.images[idx], self.labels[idx]
            if self.transform:
                image, label = self.transform([image, label])
            return image, label

class ValidationDataset(torch.utils.data.Dataset):
    """
    build validation dataset
    Note that Huggingface requires that __getitem__ method returns dict object
    """
    def __init__(self, images, labels, transform=None, huggingface=True):
        self.huggingface = huggingface

        if self.huggingface:
            self.dict = []
            for i in range(len(images)):
                temp_dict = {'image': images[i], 'label': labels[i]}
                self.dict.append(temp_dict)
        else:
            self.images = images
            self.labels = labels
            self.transform = transform

    def __len__(self):
        if self.huggingface:
            return len(self.dict)
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.huggingface:
            return self.dict[idx]
        else:
            # In ordinary Pytorch dataset, augmentation is performed within Dataset Class
            image, label = self.images[idx], self.labels[idx]
            if self.transform:
                image, label = self.transform([image, label])
            return image, label

class TestDataset(torch.utils.data.Dataset):
    """
    build test dataset
    Note that Huggingface requires that __getitem__ method returns dict object
    """
    def __init__(self, images, labels, transform=None, huggingface=True):
        self.huggingface = huggingface

        if self.huggingface:
            self.dict = []
            for i in range(len(images)):
                temp_dict = {'image': images[i], 'label': labels[i]}
                self.dict.append(temp_dict)
        else:
            self.images = images
            self.labels = labels
            self.transform = transform

    def __len__(self):
        if self.huggingface:
            return len(self.dict)
        else:
            return len(self.images)

    def __getitem__(self, idx):
        if self.huggingface:
            return self.dict[idx]
        else:
            # In ordinary Pytorch dataset, augmentation is performed within Dataset Class
            image, label = self.images[idx], self.labels[idx]
            if self.transform:
                image, label = self.transform([image, label])
            return image, label


# ------------------- model functions ---------------------

def collate_fn(examples):
    """generate batches"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# ------------------- Run Manager for no Trainer training ---------------------

class RunManager:
    """capture model stats during training"""
    def __init__(self, args):
        self.epoch_num_count = 0
        self.epoch_start_time = None
        self.args = args

        # train/validation/test metrics
        self.train_epoch_loss = 0
        self.val_epoch_loss = 0
        self.test_epoch_loss = 0

        self.run_metrics_val = None
        self.run_metrics_test = None

        # run data saves the stats from train/validation/test set
        self.run_data = []
        self.run_start_time = None

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.epoch_stats = None

    def begin_run(self, train_loader, val_loader, test_loader):
        self.run_start_time = time.time()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        logger.info('Begin Run!')

    def end_run(self, dirs):
        """save metrics from test set"""
        test_results = {f"eval_{k}": v for k, v in self.run_metrics_test.items()}
        with open(os.path.join(dirs, "test_results.json"), "w") as f:
            json.dump(test_results, f)
        self.epoch_num_count = 0
        logger.info('End Run!')

    def begin_epoch(self):
        self.epoch_num_count += 1
        self.epoch_start_time = time.time()
        # initialize metrics
        self.train_epoch_loss = 0
        self.val_epoch_loss = 0
        self.test_epoch_loss = 0
        logger.info(f'Start epoch {self.epoch_num_count}')

    def end_epoch(self):
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time
        # calculate loss
        train_loss = self.train_epoch_loss / len(self.train_loader)
        val_loss = self.val_epoch_loss / len(self.val_loader)
        logger.info(f'End epoch {self.epoch_num_count}')

        # add stats from current epoch to run data
        self.epoch_stats = OrderedDict()
        self.epoch_stats['epoch'] = self.epoch_num_count
        self.epoch_stats['train_loss'] = float(f'{train_loss:.2f}')
        self.epoch_stats['val_loss'] = float(f'{val_loss:.2f}')
        self.epoch_stats.update(self.run_metrics_val)
        self.epoch_stats['epoch_duration'] = float(f'{epoch_duration:.1f}')
        self.epoch_stats['run_duration'] = float(f'{run_duration:.1f}')
        self.run_data.append(self.epoch_stats)


    def track_train_loss(self, loss):
        # accumulate training loss for all batches
        self.train_epoch_loss += loss.item()

    def track_val_loss(self, loss):
        # accumulate validation loss for all batches
        self.val_epoch_loss += loss.item()

    def track_test_loss(self, loss):
        # accumulate test loss for all batches
        self.test_epoch_loss += loss.item()

    def collect_val_metrics(self, metric_results):
        self.run_metrics_val = {k: round(v, 3) for k, v in metric_results.items()}
        

    def collect_test_metrics(self, metric_results):
        self.run_metrics_test = {k: round(v, 3) for k, v in metric_results.items()}

    def display_epoch_results(self):
        # display stats from the current epoch
        logger.info(self.epoch_stats)

    def save(self, filename):
        pd.DataFrame.from_dict(self.run_data, orient='columns').to_csv(f'{filename}.csv')
