from transformers import ResNetForImageClassification, ResNetModel, ResNetConfig, \
    AutoModelForImageClassification, ViTForImageClassification
from peft import LoraConfig, get_peft_model

import torch.nn as nn
import logging, json

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
        # pretrained models with a classification head
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


def build_peft_model(model, peft='lora'):
    """
    Transform a model to a PEFT model. 
    https://huggingface.co/docs/peft/package_reference/peft_model
    """
    if peft == 'lora':
        config = LoraConfig(r=16, lora_alpha=16, target_modules=["query", "value"],
                            lora_dropout=0.1, bias="none", modules_to_save=["classifier"])
        peft_model = get_peft_model(model, config)

    return peft_model


# ------------------- Main Models for pipeline---------------------
def build_model(args, label2id, id2label):
    """Main function for model building"""
    # Training phase
    if not args.test:
        model = build_image_classification_model(args=args, label2id=label2id, id2label=id2label)
    
        if args.peft:
            if args.peft_config == 'lora':
                logger.info('Apply LoRA to perform peft')
                model = build_peft_model(model=model, peft=args.peft_config)
                print_trainable_parameters(model)
    # Testing phase
    else:
        model = build_image_classification_model(args=args, label2id=label2id, id2label=id2label)
    
    return model


def build_image_classification_model(args, label2id, id2label):
    """Model used in main.py"""
    # In training phase, we load pretrained model from HF
    if not args.test:
        model = AutoModelForImageClassification.from_pretrained(args.model_checkpoint,
            label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
        print_trainable_parameters(model)
    
    # In testing, we load model weights from local dir
    else:
        # If args.use_trainer is True, the model weights are saved in checkpoints, 
        # which can be found at args.best_ckpt_dir_trainer.
        # If args.use_trainer is False, the model weights are saved in arg.out_dir_no_trainer, 
        # which are saved using save_pretrained() and could be loaded using from_pretrained()
        if args.use_trainer:
            with open(args.best_ckpt_dir_trainer) as f:
                data = json.load(f)
            best_weights_path = data['best_model_checkpoint']
        else:
            best_weights_path = args.out_dir_no_trainer

        # whether we use PEFT in the training phase
        # model_id: A path to a directory containing a Lora configuration file saved using the save_pretrained().
        if args.peft:
            if args.peft_config == 'lora':
                logger.info('load LoRA peft model')
                temp_model = AutoModelForImageClassification.from_pretrained(args.model_checkpoint,
                        label2id=label2id, id2label=id2label, ignore_mismatched_sizes=True)
                model = build_peft_model(temp_model, peft=args.peft_config) 
                model = model.from_pretrained(temp_model, model_id=best_weights_path)
        else:
            model = AutoModelForImageClassification.from_pretrained(best_weights_path)

    return model
