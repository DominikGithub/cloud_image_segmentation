'''
Create pytorch dataset.
'''

from Dataloader import CloudSegDataloader
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
from transformers import Trainer
from transformers import TrainingArguments
from visualization import visualize_dataset_samples
import torch.nn as nn

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
# print(batch["pixel_values"].shape)  # (2, 3, 1024, 1024)
# print(batch["labels"].shape)        # (2, 1024, 1024)
# exit(0)


# Load base model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=1,
    ignore_mismatched_sizes=True,
)


# train model
class SegmentationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        labels = labels.unsqueeze(1).float()

        # labels von (B, 1, 1, H, W) nach (B, 1, H, W) bringen
        if labels.dim() == 5 and labels.size(2) == 1:
            labels = labels.squeeze(2)

        loss = self.criterion(logits, labels)
        return (loss, outputs) if return_outputs else loss
    

training_args = TrainingArguments(
    output_dir="./outputs/",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    push_to_hub=False,
)
trainer = SegmentationTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)
trainer.train()

