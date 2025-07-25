from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that explains unit conversions step by step. Be concise."},
            {"role": "user", "content": "How many inches are there in 2 feet?"},
            {"role": "assistant", "content": "There are 12 inches in a foot. So, 2 * 12 = <answer>24</answer>."},
            {"role": "user", "content": question}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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
