from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        messages = [
            {"role": "user", "content": "Convert 3 yards to feet."},
            {"role": "assistant", "content": "There are 3 feet in a yard.\nSo 3 yards = 3 * 3 = <answer>9</answer>."},
            {"role": "user", "content": "Convert 2 feet to inches."},
            {"role": "assistant", "content": "There are 12 inches in a foot.\nSo 2 feet = 2 * 12 = <answer>24</answer>."},
            {"role": "user", "content": question}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        print("\n=== INFERENCE PROMPT ===")
        print(prompt)

        return prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
