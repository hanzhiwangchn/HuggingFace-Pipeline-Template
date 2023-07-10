import numpy as np
import evaluate
import logging
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

from utils.utils import collate_fn


logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """
    Computes accuracy on a batch of predictions. 
    It takes a Named Tuple as input: 
        predictions, which are the logits of the model as Numpy arrays;
        label_ids, which are the ground-truth labels as Numpy arrays.
    """
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    accuracy_metric = evaluate.load("accuracy")

    preds = np.argmax(eval_pred.predictions, axis=-1)
    labels=eval_pred.label_ids
    
    all_metric_results = dict()
    all_metric_results.update(f1_metric.compute(predictions=preds, references = labels, average="weighted"))
    all_metric_results.update(precision_metric.compute(predictions=preds, references = labels, average="weighted"))
    all_metric_results.update(recall_metric.compute(predictions=preds, references = labels, average="weighted"))
    all_metric_results.update(accuracy_metric.compute(predictions=preds, references = labels))
    return all_metric_results


def build_train_argument(args):
    model_name = args.model_checkpoint.split("/")[-1]

    train_args = TrainingArguments(
        output_dir=args.out_dir + f'/{model_name}',
        logging_dir=args.log_dir + f'/{model_name}',
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        logging_strategy='epoch',
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args.batch_size,
        fp16=args.cuda,
        num_train_epochs=args.num_train_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        label_names=["labels"],
        save_total_limit = 2,
    )
    return train_args


def build_trainer(model, train_args, dataset_train, dataset_val, image_processor):
    """build trainer"""
    trainer = Trainer(model=model, args=train_args, data_collator=collate_fn,
                      train_dataset=dataset_train, eval_dataset=dataset_val,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                      tokenizer=image_processor, compute_metrics=compute_metrics)
    return trainer

def train_val_test(trainer, dataset_test, args):
    """
    Trainer.train() and .evaluate() will use train_dataset and eval_dataset by default. 
    While Trainer.predict() asks for a dataset and produce metrics if labels are provided.
    "https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer"

    All performance will be saved in model_ckpt_results folder and stdout should be ignored.
    """
    if not args.test:
        logger.info('--- Training started ---')
        train_results = trainer.train()
        
        # save model and results
        trainer.save_model()
        trainer.log_metrics("train", train_results.metrics)
        trainer.save_metrics("train", train_results.metrics)
        trainer.save_state()

        logger.info('--- Training ended ---')

        logger.info('--- Evaluation started ---')
        # evaluate on the validation set
        metrics = trainer.evaluate()

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        logger.info('--- Evaluation ended ---')

    logger.info('--- Testing started ---')
    # trainer.predict() returns logits (num_instance, num_class); 
    # target label (num_instance) and corresponding metrics
    # Since we load the best model using "load_best_model_at_end=True", best trainer has been loaded.
    prediction = trainer.predict(dataset_test)

    trainer.log_metrics("test", prediction[2])
    trainer.save_metrics("test", prediction[2])

    logger.info('--- Testing ended ---')
