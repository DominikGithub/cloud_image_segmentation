'''
Pytorch Segformer model fine tuning for cloud segmentation task.
'''

from Dataloader import CloudSegDataloader

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
from visualization import visualize_dataset_samples
import torch.nn.functional as F
import torch.nn as nn
import torch

BATCH_SIZE = 20

# hf segformer data preprocessor
preprocessor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", use_fast=False)
preprocessor.do_resize = False
preprocessor.do_rescale = False
preprocessor.do_normalize = True  

# load data
train_ds = CloudSegDataloader('training', preprocessor)
val_ds = CloudSegDataloader('validation', preprocessor)
print(len(train_ds), len(val_ds))

# ## plot samples to validate data loading
# visualize_dataset_samples(val_ds, 2)
# exit(0)

# ## check dataloader data shapes
# from torch.utils.data import DataLoader
# dl = DataLoader(train_ds, batch_size=2)
# batch = next(iter(dl))
# print(batch)
# print(batch["pixel_values"].shape)  # (2, 3, 1024, 1024)
# print(batch["labels"].shape)        # (2, 1024, 1024)
# exit(0)


# Load base model
base_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=1,
    ignore_mismatched_sizes=True,
)
# freeze base model
for param in base_model.segformer.parameters():
    param.requires_grad = False
# for param in base_model.decode_head.parameters():
#     param.requires_grad = False
# # print model layer training state
# for name, param in base_model.named_parameters():
#     print(f"{name}: requires_grad={param.requires_grad}")


class SegformerWithUpsample(torch.nn.Module):
    def __init__(self, base_model, target_size=(1024, 1024)):
        super().__init__()
        self.base_model = base_model
        self.target_size = target_size

    def forward(self,pixel_values=None,labels=None,**kwargs):
        outputs = self.base_model(pixel_values=pixel_values, labels=labels, **kwargs)
        logits = outputs.logits           # [B, C, H_out, W_out]
        # upscaling
        logits = F.interpolate(
            logits,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )

        # loss recalculation (for BCE/CE in 1024x1024)
        if labels is not None:
            # labels shape (B,H,W) → (B,C,H,W)
            if labels.ndim == 3:
                labels_for_loss = labels.unsqueeze(1).float()
            else:
                labels_for_loss = labels.float()
            loss = F.binary_cross_entropy_with_logits(logits, labels_for_loss)
        else:
            loss = None
        return SemanticSegmenterOutput(
            loss      = loss,
            logits    = logits,
            hidden_states = outputs.hidden_states,
            attentions    = outputs.attentions,
        )


model = SegformerWithUpsample(base_model)

# train model
class SegmentationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        pixel_values = inputs["pixel_values"]
        labels = inputs["labels"]
        
        outputs = model(pixel_values)
        logits = outputs.logits
        # Upsampling
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss = self.criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss
    

def compute_metrics(eval_pred):
    '''
    Validation metric logging
    '''
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    mean_iou = np.mean(preds == labels)
    return {
        "accuracy": accuracy_score(labels.flatten(), preds.flatten()),
        "f1": f1_score(labels.flatten(), preds.flatten(), average="macro"),
        "mean_iou": mean_iou,
    }


class LogCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print("Evaluation metrics:", metrics)


training_args = TrainingArguments(
    output_dir="./outputs/",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,
    # logging
    logging_dir='./outputs/pytorch_logs',
    logging_strategy="epoch", # steps
    logging_steps=50,
    # validation
    eval_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_steps=100,
    load_best_model_at_end=True,  
    metric_for_best_model="f1",  # eval_loss, mean_iou, f1
    greater_is_better=True,

)
trainer = SegmentationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5), LogCallback()],
)
trainer.train()


# final evaluation metrics
metrics = trainer.evaluate()
print(metrics)

# # Save model
# file_path = './segformer_cloud.pth'
# model.save_pretrained(file_path)
# # torch.save(model.state_dict(), file_path)

