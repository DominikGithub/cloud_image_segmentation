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
import time

TIME_SEC = int(time.time())
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
for param in base_model.segformer.parameters(): # encoder
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
# class BinaryIoULoss(nn.Module):
#     def __init__(self, eps=1e-6):
#         super().__init__()
#         self.eps = eps

#     def forward(self, logits, targets):
#         probs = torch.sigmoid(logits)
#         preds = (probs > 0.5).float()
#         targets = targets.float()

#         intersection = (preds * targets).sum(dim=(1, 2, 3))
#         union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection

#         iou = (intersection + self.eps) / (union + self.eps)
#         loss = 1.0 - iou  # IoU loss = 1 - IoU score
#         return loss.mean()

# class BCEAndIoULoss(nn.Module):
#     def __init__(self, alpha=0.5, pos_weight=4.0):
#         super().__init__()
#         self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
#         self.iou = BinaryIoULoss()
#         self.alpha = alpha

#     def forward(self, logits, targets):
#         return self.alpha * self.bce(logits, targets.float()) + (1 - self.alpha) * self.iou(logits, targets)



class SegmentationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = BCEAndIoULoss()

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        pixel_values = inputs["pixel_values"]
        labels = inputs["labels"]
        
        outputs = model(pixel_values)
        logits = outputs.logits
        # Upsampling
        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss = self.criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss
    

def compute_metrics(eval_pred, threshold=0.5):
    '''
    Validation metric logging
    '''
    logits, labels = eval_pred
    preds = logits.argmax(axis=0)
    
    prob   = torch.sigmoid(torch.from_numpy(logits))
    preds  = (prob > threshold).long()
    labels = torch.from_numpy(labels).long()

    tp = (preds & labels).sum()
    fp = (preds & (~labels)).sum()
    fn = ((~preds) & labels).sum()
    iou = tp / (tp + fp + fn + 1e-6)
    return {
        'acc': accuracy_score(labels.flatten(), preds.flatten()),
        'f1': f1_score(labels.flatten(), preds.flatten(), average="weighted"), # weighted micro
        "b_iou": iou,
    }


class LogCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print("Validation set:", metrics)


def train_model():
    '''
    Train the model
    '''
    training_args = TrainingArguments(
        output_dir="./outputs/",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=1,
        # logging
        # logging_dir='./outputs/pytorch_logs/',                  # NOTE <- not working
        logging_strategy="epoch",
        logging_steps=1_000_00,
        # validation
        eval_steps=1000,
        eval_strategy="epoch",
        save_strategy="best",
        save_steps=500,
        load_best_model_at_end=True,  
        metric_for_best_model="f1",  # iou, f1
        greater_is_better=True,

    )
    trainer = SegmentationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10), LogCallback()],
    )
    trainer.train()


    # final evaluation metrics
    metrics = trainer.evaluate()
    print('Final validation metric:', metrics)

    # Save model
    file_path = f'./segformer_cloud_{TIME_SEC}.pth'
    torch.save(model.state_dict(), file_path)



if __name__ == "__main__":
    train_model()