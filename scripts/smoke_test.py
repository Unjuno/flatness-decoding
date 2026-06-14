from flatness_decoding.decoding import generate_flatness_controlled
from flatness_decoding.model_io import load_causal_lm


def main() -> None:
    model, tokenizer, device = load_causal_lm(
        model_name="distilgpt2",
        device_name="auto",
        dtype_name="auto",
    )

    text, records = generate_flatness_controlled(
        model=model,
        tokenizer=tokenizer,
        prompt="Once upon a time",
        max_new_tokens=30,
        n_trials=4,
        proposal_temperature=1.0,
        method="minflat",
        device=device,
        seed=0,
    )

    print(f"device: {device}")
    print("--- generated text ---")
    print(text)
    print("--- first step record ---")
    print(records[0].to_dict() if records else None)


if __name__ == "__main__":
    main()
