from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, Seq2SeqTrainer
from datasets import load_dataset

data = load_dataset("opus_books", "en-fr")
data = data["train"].train_test_split(test_size=0.2)
print(data)

prefix = "translate English to Morse: "

checkpoint="google-t5/t5-small"
# Tokenization
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

def dataGeneration(examples):
    input = [prefix + example["en"] for example in examples["translation"]]
    target = [text_to_morse(example["en"]) for example in examples["translation"]]
    tokenized = tokenizer(input, text_target=target, max_length=128, truncation=True)
    return tokenized



print("Start...")
tokenized_data = data.map(dataGeneration, batched=True)
print(tokenized_data["train"][0])

data_collator = DataCollatorForSeq2Seq(tokenizer, model=checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="t5morse",
    learning_rate=2e-5,
    eval_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    num_train_epochs=5,
    predict_with_generate=True,
)


trainer = Seq2SeqTrainer(
    model = model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()