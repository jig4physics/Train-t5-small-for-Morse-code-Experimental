from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, Seq2SeqTrainer
from datasets import load_dataset
from utils.Text2Morse import Text2Morse

class Trainer:
    def __init__(self):
        self.data  = load_dataset("opus_books", "en-fr")
        self.data = self.data["train"].train_test_split(test_size=0.2)
        self.prefix = "translate English to Morse: "
        self.checkpoint="google-t5/t5-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)
        self.T2M = Text2Morse()
    
    def dataGeneration(self, examples):
        input = [self.prefix + example["en"] for example in examples["translation"]]
        target  = [self.T2M.text_to_morse(example["en"] for example in examples["translation"])]
        tokenized = self.tokenizer(input, text_target=target, max_length=128, truncation=True)
        return tokenized

    def train(self):
        print("Training START")
        tokenized_data = data.map(self.dataGeneration, batched=True)
        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.checkpoint)


if __name__ == "__main__":
    t = Trainer()
    t.train()