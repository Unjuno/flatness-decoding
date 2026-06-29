# Tokenizer / Model Compatibility

- requested_model_name: `google/gemma-3-4b-it`
- tokenizer.name_or_path: `google/gemma-3-4b-it`
- model.config.name_or_path: `google/gemma-3-4b-it`
- tokenizer class: `GemmaTokenizer`
- model class: `Gemma3ForConditionalGeneration`
- tokenizer vocab_size: `262144`
- len(tokenizer): `262145`
- model input embedding vocab size: `262208`
- model output head vocab size: `262208`
- pad/eos/bos token ids: `0` / `1` / `2`
- chat_template exists: `True`
- vocab compatible: `False`
- cache path warning: `None`
- model error: `None`

## Smoke Tests
- `Hello.` -> roundtrip `True` (2 tokens)
- `[ WP ] Write a road trip story.` -> roundtrip `True` (9 tokens)
- `Continue the following WritingPrompts story prompt as narrative prose.` -> roundtrip `True` (12 tokens)

## Generation Smoke
- plain causal prompt: `Hello.`
- generated text: `Hello.Hello.Hello.Hello.Hello.Hello.Hello.Hello.Hello.Hello.Hello.Hello.Hello.Hello.Hello.Hello.Hello.`

## Chat Template Smoke
- generated text: `user
Write one short sentence about a traveler at a train station.
model
The weary traveler slumped onto a bench, clutching his ticket and watching the train pull in.`
- recommendation: Qwen-style chat_template exists; consider a separate chat-template condition.
