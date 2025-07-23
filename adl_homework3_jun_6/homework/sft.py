from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel
    from transformers import AutoTokenizer

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name
    print("model_path in load:", model_path)

    llm = BaseLLM()
    # Load tokenizer from same path as LoRA adapters
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    llm.tokenizer = tokenizer

    llm.model = PeftModel.from_pretrained(llm.model, str(model_path), local_files_only=True).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(question, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    full["labels"] = input_ids.copy()  # Supervise entire sequence
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    messages = [
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": f"<answer>{round(float(answer), 2)}</answer>"}
    ]
    tokenizer = BaseLLM().tokenizer
    full_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    full_prompt += tokenizer.eos_token

    print("\n=== TRAINING PROMPT ===")
    print(full_prompt)

    return {"question": full_prompt, "answer": ""}

class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formatted_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formatted_data)


def train_model(output_dir: str, **kwargs):
    from peft import get_peft_model, LoraConfig, TaskType
    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

    dataset = Dataset("train")
    llm = BaseLLM()
    tokenizer = llm.tokenizer

    train_ds = TokenizedDataset(tokenizer, dataset, format_example)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none"
    )
    llm.model = get_peft_model(llm.model, peft_config)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        logging_steps=10,
        save_steps=100,
        save_total_limit=1,
        learning_rate=5e-5,
        fp16=False,
        report_to="none"
    )

    trainer = Trainer(
        model=llm.model,
        args=args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=llm.model, padding=True)
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)  # ⬅️ Saves tokenizer for aligned evaluation

    test_model(output_dir)


def test_model(ckpt_path: str):
    from pathlib import Path
    from peft import PeftModel
    from transformers import AutoTokenizer

    testset = Dataset("valid")
    model_path = Path(ckpt_path)
    print("model_path in test:", model_path)

    llm = BaseLLM()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
    llm.tokenizer = tokenizer

    llm.model = PeftModel.from_pretrained(llm.model, str(model_path), local_files_only=True).to(llm.device)
    llm.model.eval()

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
