# Tokenizer / Model Compatibility

- requested_model_name: `Qwen/Qwen2.5-1.5B-Instruct`
- tokenizer.name_or_path: `Qwen/Qwen2.5-1.5B-Instruct`
- model.config.name_or_path: `None`
- tokenizer class: `Qwen2Tokenizer`
- model class: `None`
- tokenizer vocab_size: `151643`
- len(tokenizer): `151665`
- model input embedding vocab size: `None`
- model output head vocab size: `None`
- pad/eos/bos token ids: `151643` / `151645` / `None`
- chat_template exists: `True`
- vocab compatible: `True`
- cache path warning: `None`
- model error: `Qwen/Qwen2.5-1.5B-Instruct does not appear to have a file named pytorch_model.bin or model.safetensors.`

## Smoke Tests
- `Hello.` -> roundtrip `True` (2 tokens)
- `[ WP ] Write a road trip story.` -> roundtrip `True` (9 tokens)
- `Continue the following WritingPrompts story prompt as narrative prose.` -> roundtrip `True` (12 tokens)

## Generation Smoke
- plain causal prompt: `Hello.`
- generated text: `None`

## Chat Template
- recommendation: Qwen-style chat_template exists; consider a separate chat-template condition.
