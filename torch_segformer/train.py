'''
Pytorch Segformer model transfer learning for cloud segmentation task.

MlFlow server: 
$ export MLFLOW_FLASK_SERVER_SECRET_KEY="secret"
$ export MLFLOW_TRACKING_USERNAME="secret"
$ export MLFLOW_TRACKING_PASSWORD="secret"

$ mlflow server --host 0.0.0.0 --port 8080 [--app-name basic-auth]
'''

from Dataloader import CloudSegDataloader

import numpy as np
from sklearn.metrics import f1_score
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback
import torch.nn.functional as F
import torch.nn as nn
import torch
import time
import csv
import os
import mlflow

# MLflow experiment tacking
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.config.enable_async_logging(True)  
os.environ["MLFLOW_ASYNC_LOGGING_BUFFERING_SECONDS"] = "2"


TIME_SEC = int(time.time())
BATCH_SIZE = 64


# segformer data preprocessor
preprocessor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512", use_fast=False)
preprocessor.do_resize = False
preprocessor.do_rescale = False
preprocessor.do_normalize = True  

# load data
train_ds = CloudSegDataloader('training', preprocessor)
val_ds = CloudSegDataloader('validation', preprocessor)
test_ds = CloudSegDataloader('test', preprocessor)
n_steps_per_epoch = len(train_ds)
print('# batches (train, val, test):', n_steps_per_epoch, len(val_ds), len(test_ds))


# Load base model
base_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=1,
    ignore_mismatched_sizes=True,
)
# freeze base model encoder layers
for param in base_model.segformer.parameters():
    param.requires_grad = False
# for param in base_model.decode_head.parameters():
#     param.requires_grad = True
# # print model layer training state
# for name, param in base_model.named_parameters():
#     print(f"{name}: requires_grad={param.requires_grad}")


class SegformerWithUpsample(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        # upsampling layer
        self.refine1 = nn.Sequential(             # 256 -> 512
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.refine2 = nn.Sequential(             # 512 -> 1024
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, pixel_values=None, labels=None, **kwargs):
        out = self.base(pixel_values=pixel_values, **kwargs)
        x   = out.logits                         # (B,1,256,256)
        x   = self.refine1(x)                    # (B,32,512,512)
        logits = self.refine2(x)                 # (B,1,1024,1024)

        loss = None
        if labels is not None:
            if labels.ndim == 3:
                labels = labels.unsqueeze(1).float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        return SemanticSegmenterOutput(
            loss=loss, logits=logits,
            hidden_states=out.hidden_states, attentions=out.attentions
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
    
    def log_mlflow(self, logs):
        super().log(logs)
        if self.args.local_rank in [-1, 0]:
            for k, v in logs.items():
                mlflow.log_metric(k, v, step=self.state.global_step)
                
    def evaluate_on_train(self):
        # calculate metrics on training data for mlflow
        train_preds = self.predict(self.train_dataset)
        preds = train_preds.predictions
        labels = train_preds.label_ids
        # iou
        prob   = torch.sigmoid(torch.from_numpy(preds))
        preds  = (prob > 0.5).long()
        labels = torch.from_numpy(labels).long()
        tp = (preds & labels).sum()
        fp = (preds & (~labels)).sum()
        fn = ((~preds) & labels).sum()
        iou = (tp / (tp + fp + fn + 1e-6)).cpu().numpy().astype(np.float32)
        # f1
        preds = (preds > 0.5).cpu().numpy().astype(np.float32)
        f1 = f1_score(labels.flatten(), preds.flatten())
        self.log_mlflow({"train_f1": f1, 'train_iou': float(iou)})
    

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
    f1 = f1_score(labels.flatten(), preds.flatten(), average="weighted") # micro
    return {
        'f1': f1,
        "iou": iou,
    }


class MlFlowLoggerCb(TrainerCallback):
    def __init__(self, trainer=None):
        self.trainer = trainer
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if control.should_log:
            mlflow.log_params(metrics)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer:
            self.trainer.evaluate_on_train()


class FileLoggerCb(TrainerCallback):
    def __init__(self, filename=f"./metrics_{TIME_SEC}"):
        self.filename = filename

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            # csv
            with open(self.filename+'.csv', 'a') as fd:
                w = csv.DictWriter(fd, logs.keys())
                if logs['epoch'] == 1.0:
                    w.writeheader()
                w.writerow(logs)
            # json
            with open(self.filename+'.log', "a") as f:
                f.write(f"{logs}\n")
                

def train_model():
    '''
    Train the model
    '''
    # configure tracking 
    mlflow.set_experiment(f"Experiment {TIME_SEC}")
    with mlflow.start_run():
        training_args = TrainingArguments(
            output_dir=f"./outputs/{TIME_SEC}/",
            learning_rate=2e-5,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            num_train_epochs=150,  
            logging_strategy="epoch",
            logging_steps=n_steps_per_epoch,
            eval_steps=n_steps_per_epoch,
            eval_strategy="epoch",
            save_strategy="best",
            save_steps=n_steps_per_epoch,
            load_best_model_at_end=True,  
            metric_for_best_model="iou",  # iou, f1
            greater_is_better=True,

        )
        trainer = SegmentationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10), FileLoggerCb()]
        )
        # add callback with trainer ref after trainer init
        callback = MlFlowLoggerCb(trainer=trainer)
        trainer.add_callback(callback)
        # train model
        try: trainer.train()
        except KeyboardInterrupt: print('Stop training.')

        # final evaluation metrics
        metrics = trainer.evaluate()
        print('Final validation metric:', metrics)

    # Save model
    file_path = f'./segformer_cloud_{TIME_SEC}.pth'
    torch.save(model.state_dict(), file_path)


if __name__ == "__main__":
    train_model()