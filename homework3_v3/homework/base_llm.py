from typing import overload
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        You don't need to change this function for now.
        """
        return question



    def parse_answer(self, output_text: str):
        """
        Extract the first number (int or float) from the model's output.
        """
        try:
            matches = re.findall(r"-?\d+(?:\.\d+)?", output_text)
            if matches:
                return float(matches[0])
            else:
                return None
        except Exception as e:
            print("⚠️ Error in parse_answer:", e)
            return None


    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        """
        Generate a single output string from a prompt.
        """
        self.model.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

    def batched_generate(
            self,
            prompts: list[str],
            num_return_sequences: int | None = None,
            temperature: float = 0,
            max_new_tokens: int = 64
    ) -> list[str] | list[list[str]]:
        from tqdm import tqdm

        self.tokenizer.padding_side = "left"
        self.model.eval()

        # Prevent OOM issues
        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches {micro_batch_size}"
                )
                for r in self.batched_generate(prompts[idx : idx + micro_batch_size], num_return_sequences, temperature)
            ]

        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)

        do_sample = temperature > 0
        num_return_sequences = num_return_sequences or 1

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Remove the prompt from the output
        generated_tokens = outputs[:, inputs["input_ids"].shape[1]:]

        decoded = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        if num_return_sequences == 1:
            return decoded
        else:
            return [decoded[i:i + num_return_sequences] for i in range(0, len(decoded), num_return_sequences)]

    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
