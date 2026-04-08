from llama_cpp import Llama

# Download Q8_0 GGUF (~4GB) from HuggingFace:
# bartowski/Phi-3.5-mini-instruct-GGUF
llm = Llama.from_pretrained(
    repo_id="bartowski/Phi-3.5-mini-instruct-GGUF",
    filename="Phi-3.5-mini-instruct-Q8_0.gguf",  # INT8
    # filename="Phi-3.5-mini-instruct-Q4_K_M.gguf",  # INT4 if still tight
    n_ctx=4096,
    n_threads=8,
)

output = llm("if an infant and a pregnant woman fall into a well, and i can only save only one of them, whom should i save?", max_tokens=256)
print(output["choices"][0]["text"])
