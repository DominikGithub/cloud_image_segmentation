'''
Create pytorch dataset.
'''

from Dataloader import CloudSegDataloader
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import Trainer
from transformers import TrainingArguments
from visualization import visualize_dataset_samples

# load data
train_ds = CloudSegDataloader('training')
val_ds = CloudSegDataloader('validation')
print(len(train_ds), len(val_ds))

# plot samples to validate data loading
# visualize_dataset_samples(val_ds, 1)


# Load model 
processor = AutoImageProcessor.from_pretrained("nvidia/mit-b0", use_fast=False)
model = AutoModelForImageClassification.from_pretrained("nvidia/mit-b0")
# from transformers import pipeline
# pipeline = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf", device="cuda")


# train model
training_args = TrainingArguments(
    output_dir="./outputs/",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    push_to_hub=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=processor,
    # data_collator=data_collator,
)
trainer.train()

