from transformers import ResNetForImageClassification, ResNetModel, \
    AutoModelForImageClassification, ViTForImageClassification, ResNetConfig, ViTModel
from peft import LoraConfig, get_peft_model

import torch.nn as nn
import logging
import json, os

logger = logging.getLogger(__name__)

# For image classification task, we will use pretrained ViT model with LoRA PEFT
# HF models could be used as normal Pytorch models ,see "Customized Model with PreTrained HF ResNet"


# ------------------- Customized Model with PreTrained HF ResNet ---------------------
class ResNetModelHF(nn.Module):
    """
    PreTrained ResNet model without classification head + custom regression head
    """
    def __init__(self):
        super(ResNetModelHF, self).__init__()
        self.base_model = ResNetModel.from_pretrained('microsoft/resnet-50')
        # base model with output shape (*, 2048, 7, 7)
        self.custom_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.5),
            nn.Linear(2048 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values)
        # outputs return an HF module, so we extract the values from it
        # the values could be extracted using outputs.logits or outputs[0]
        outputs = self.custom_head(outputs.logits)
        return outputs


class ResNetModelWithHeadHF(nn.Module):
    """
    PreTrained ResNet model with classification head + custom regression head
    """
    def __init__(self):
        super(ResNetModelWithHeadHF, self).__init__()
        self.base_model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')
        # base model with output shape (*, 1000)
        self.custom_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1000, 1),
        )

    def forward(self, pixel_values):
        outputs = self.base_model(pixel_values)
        outputs = self.custom_head(outputs.logits)
        return outputs


# ------------------- PreTrained HF ViT ---------------------
def load_ViT_model():
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
    return model


# ------------------- Build models in General ---------------------
def build_resnet(options):
    if options == 'from_config':
        # Initializing a ResNet resnet-50 style configuration
        configuration = ResNetConfig()
        # Initializing a model (with random weights) from the resnet-50 style configuration
        resnet_model = ResNetModel(configuration)
    elif options == 'pretrained':
        # pretrained models will have a specific structure
        resnet_model = ResNetModel.from_pretrained('microsoft/resnet-50')
    elif options == 'pretrained_with_head':
        resnet_model = ResNetForImageClassification.from_pretrained('microsoft/resnet-50')

    return resnet_model


# ------------------- LoRA PEFT ---------------------
def print_trainable_parameters(model):
    """Prints the number of trainable parameters in the model."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(f"trainable params: {trainable_params} || all params: {all_param} || "
                f"trainable%: {100 * trainable_params / all_param:.2f}")


def build_id_label_mapping(dataset):
    labels = dataset.features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = i
        id2label[i] = label

    return label2id, id2label


def build_image_classification_model(args, label2id, id2label):
    """Model used in main.py"""
    if not args.test:
        # Load pretrained model from HF
        model = AutoModelForImageClassification.from_pretrained(args.model_checkpoint,
            label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
        print_trainable_parameters(model)
    else:
        # Load fine-tuned model from local dir for testing.
        # When we use trainer, the model weights are saved in checkpoints, 
        # which can be found at args.best_model_ckpt_dir.
        # When we do not use trainer, the model weights are saved in 
        if args.use_trainer:
            with open(args.best_model_ckpt_dir) as f:
                data = json.load(f)
            if args.peft:
                if args.peft_config == 'lora':
                    temp_model = AutoModelForImageClassification.from_pretrained(args.model_checkpoint,
                        label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
                    model = build_peft_model(temp_model, peft='lora')
                    model = model.from_pretrained(temp_model, model_id=data['best_model_checkpoint'])
            else:
                model = AutoModelForImageClassification.from_pretrained(data['best_model_checkpoint'])
        else:
            if args.peft:
                if args.peft_config == 'lora':
                    temp_model = AutoModelForImageClassification.from_pretrained(args.model_checkpoint,
                        label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
                    model = build_peft_model(temp_model, peft='lora')
                    model = model.from_pretrained(temp_model, model_id=args.out_dir_no_trainer)
            else:
                model = AutoModelForImageClassification.from_pretrained(args.out_dir_no_trainer)

    return model


def build_peft_model(model, peft='lora'):
    """PEFT Model used in main.py"""
    if peft == 'lora':
        config = LoraConfig(r=16, lora_alpha=16, target_modules=["query", "value"],
                            lora_dropout=0.1, bias="none", modules_to_save=["classifier"])
        peft_model = get_peft_model(model, config)

    return peft_model

def build_model(args, label2id, id2label):
    """Whole function to build the model"""
    if not args.test:
        model = build_image_classification_model(args=args, label2id=label2id, id2label=id2label)
    
        if args.peft:
            if args.peft_config == 'lora':
                logger.info('Apply LoRA to perform peft')
                model = build_peft_model(model=model, peft='lora')
                print_trainable_parameters(model)
    else:
        model = build_image_classification_model(args=args, label2id=label2id, id2label=id2label)
    
    return model
