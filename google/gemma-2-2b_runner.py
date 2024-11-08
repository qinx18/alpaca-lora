# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.prompter import Prompter

tokenizer = AutoTokenizer.from_pretrained("/home/qinxiao/KV-cache-testing/alpaca-lora/google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained(
    "/home/qinxiao/KV-cache-testing/alpaca-lora/google/gemma-2-2b",
    device_map="cuda:1",
)

prompter = Prompter(template_name="alpaca", verbose=True)
prompt = prompter.generate_prompt(
        instruction = "Write codes of quicksort algorithms in python, c++ and java.",
        input = ""
)
prompt = [prompt,] * 8
input_text = prompt
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda:1")

outputs = model.generate(**input_ids, max_new_tokens=2000, cache_implementation = "offloaded_static",)

print(tokenizer.decode(outputs[0]))
