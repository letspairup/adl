import math

from .base_llm import BaseLLM
from .sft import test_model
from torch.optim import AdamW


def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(output_dir: str = "/Users/models/rft_model", num_train_epochs: int = 3, batch_size: int = 4):
    import torch
    from peft import get_peft_model, LoraConfig, TaskType
    from .data import Dataset, is_answer_valid

    llm = BaseLLM()
    model = llm.model
    tokenizer = llm.tokenizer
    model.train()

    # Apply LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.to(llm.device)

    dataset = Dataset("train") # Load the training dataset
    #dataset = Dataset("train")[:100] # Limit to 100 samples for faster training
    optimizer = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(num_train_epochs):
        print(f"🌍 Starting RFT Epoch {epoch+1}")
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            losses = []

            for question, gold_answer in batch:
                prompt = question
                encoded = tokenizer(prompt, return_tensors="pt", padding=True).to(llm.device)
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                # Step 1: Generate prediction (no grad)
                with torch.no_grad():
                    output_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=64, do_sample=False)
                    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    predicted_answer = llm.parse_answer(output_text)

                # Debug output
                print("🔍 Prompt:", prompt)
                print("🧠 Predicted:", predicted_answer)
                print("🎯 Gold:", gold_answer)

                # Step 2: Compute reward
                reward = 1.0 if is_answer_valid(predicted_answer, gold_answer) else 0.0
                print("✅ Reward:", reward)
                print(f"🧪 Raw output: {repr(output_text)} | Parsed: {predicted_answer}")

                if (
                        predicted_answer is None
                        or (isinstance(predicted_answer, str) and predicted_answer.strip() == "")
                        or (isinstance(predicted_answer, float) and math.isnan(predicted_answer))
                ):

                    print("⚠️ Warning: Empty prediction, skipping...")
                    continue

                # Step 3: Recompute logits for grad-based loss
                gen_tokens = output_ids[:, input_ids.shape[1]:]
                if gen_tokens.shape[1] == 0:
                    print("⚠️ Skipping empty generation")
                    continue

                full_input = output_ids[:, :-1]  # input to predict next token
                target = output_ids[:, 1:]       # target is next token

                outputs = model(full_input)
                logits = outputs.logits
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                selected = log_probs.gather(2, target.unsqueeze(-1)).squeeze(-1)
                selected_gen = selected[:, -gen_tokens.shape[1]:]  # only generated part

                log_prob = selected_gen.mean()
                loss = -log_prob * reward
                losses.append(loss)

            if losses:
                total_loss = torch.stack(losses).mean()
                print(f"📉 Total Loss: {total_loss.item():.4f}")
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    print(f"💾 Saving model to {output_dir}")
    model.save_pretrained(output_dir)

    print("✅ Done. Testing RFT model...")
    test_model(output_dir)


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})
