from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Formats the question using a chain-of-thought few-shot prompt and chat template.
        """
        few_shot = [
            {
                "role": "user",
                "content": "Convert 3 meters to feet."
            },
            {
                "role": "assistant",
                "content": "1 meter = 3.28084 feet. So, 3 * 3.28084 = <answer>9.84252</answer>"
            },
            {
                "role": "user",
                "content": "Convert 5 miles to kilometers."
            },
            {
                "role": "assistant",
                "content": "1 mile = 1.60934 kilometers. So, 5 * 1.60934 = <answer>8.0467</answer>"
            },
            {
                "role": "user",
                "content": question
            }
        ]

        return self.tokenizer.apply_chat_template(few_shot, tokenize=False)



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
