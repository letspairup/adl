from homework.sft import BaseLLM
from peft import PeftModel
from transformers import AutoTokenizer

model_path = "/Users/sft_model"

llm = BaseLLM()
llm.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
llm.model = PeftModel.from_pretrained(llm.model, model_path, local_files_only=True).to(llm.device)
llm.model.eval()

# Manually construct the same prompt style as training (chat-style)
messages = [
    {"role": "user", "content": "Convert 3 yards to feet."}
]
prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False)

print("\n=== MODEL GENERATION ===")
print(llm.generate(prompt))
