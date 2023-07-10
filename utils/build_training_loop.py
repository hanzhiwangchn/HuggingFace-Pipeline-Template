import os, math, logging

import torch
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
import evaluate
import json

from utils.utils import collate_fn
from utils.build_models import build_model

logger = logging.getLogger(__name__)


def build_data_loader(args, dataset_train, dataset_val, dataset_test):
    """make data loader"""
    # build dataloader configurations
    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'collate_fn': collate_fn}
    validation_kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'collate_fn': collate_fn}
    test_kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'collate_fn': collate_fn}
    if torch.cuda.is_available():
        cuda_kwargs = {'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        validation_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # initialize loader
    train_loader = torch.utils.data.DataLoader(dataset_train, **train_kwargs)
    val_loader = torch.utils.data.DataLoader(dataset_val, **validation_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    return train_loader, val_loader, test_loader


def build_optimizer(model, train_loader, args):
    optimizer = AdamW(model.parameters(), lr=args.lr)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    return optimizer, lr_scheduler


def train_val_test_pt(args, train_loader, val_loader, test_loader, model, device, 
    optimizer, lr_scheduler, m, label2id, id2label):
        # get progress bar
    if not args.test:
        progress_bar = tqdm(range(args.max_train_steps))
        best_loss = 100

        # train and evaluate
        for epoch in range(args.num_train_epochs):
            m.begin_epoch()
            train(model, train_loader, optimizer, lr_scheduler, device, progress_bar, m)
            validation(model, val_loader, epoch, device, m)
            m.end_epoch()
            m.display_epoch_results()

            # save model
            if m.epoch_stats['val_loss'] < best_loss:
                logger.info(f'Lower validation loss found at epoch {m.epoch_num_count}')
                best_loss = m.epoch_stats['val_loss']
                model.save_pretrained(args.out_dir_no_trainer)
    
    # testing
    # TODO: testing should be done using the best model

    test(model, test_loader, device, m, args, label2id, id2label)

    m.end_run(dirs=args.out_dir_no_trainer)

    # save stats
    m.save(os.path.join(args.out_dir_no_trainer, f'{args.model_config}_runtime_stats'))
        

def train(model, train_loader, optimizer, lr_scheduler, device, progress_bar, m):
    model.train()
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        m.track_train_loss(loss=loss)
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


def validation(model, val_loader, epoch, device, m):
    all_metrics = dict()
    all_metric_type = ["accuracy", "recall", "precision", "f1"]
    for metric in all_metric_type:
        all_metrics[metric] = evaluate.load(metric)
    
    all_metrics_results = dict()

    model.eval()
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            m.track_val_loss(loss=outputs.loss)

            predictions = outputs.logits.argmax(dim=-1)
            for metric in all_metric_type:
                all_metrics[metric].add_batch(predictions=predictions, references=batch["labels"])

    for metric in all_metric_type:
        if metric in ["recall", "precision", "f1"]:
            all_metrics_results.update(all_metrics[metric].compute(average='weighted'))
        else:
            all_metrics_results.update(all_metrics[metric].compute())
    
    # track val metrics
    m.collect_val_metrics(metric_results=all_metrics_results)


def test(model, test_loader, device, m, args, label2id, id2label):
    all_metrics = dict()
    all_metric_type = ["accuracy", "recall", "precision", "f1"]
    for metric in all_metric_type:
        all_metrics[metric] = evaluate.load(metric)
    
    all_metrics_results = dict()

    # TODO: Here is a bit weild. When we do training, we need to load model with best accuracy for testing
    # when we do testing, the model with best accuracy has already been loaded
    if not args.test:
        model = build_model(args, label2id, id2label)
        if args.peft:
            if args.peft_config == 'lora':
                model.from_pretrained(model, model_id=args.out_dir_no_trainer)
        else:
            model = model.from_pretrained(args.out_dir_no_trainer)
        model.to(device)

    model.eval()
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            m.track_test_loss(loss=outputs.loss)

            predictions = outputs.logits.argmax(dim=-1)

            for metric in all_metric_type:
                all_metrics[metric].add_batch(predictions=predictions, references=batch["labels"])

    for metric in all_metric_type:
        if metric in ["recall", "precision", "f1"]:
            all_metrics_results.update(all_metrics[metric].compute(average='weighted'))
        else:
            all_metrics_results.update(all_metrics[metric].compute())

    # track test metrics
    m.collect_test_metrics(metric_results=all_metrics_results)

    # TODO: only used in testing
    with open(os.path.join(args.out_dir_no_trainer, "test_results_add.json"), "w") as f:
        json.dump(all_metrics_results, f)
