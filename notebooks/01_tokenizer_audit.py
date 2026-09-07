from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


# Tokenizer candidates to compare
CANDIDATES = {
    "bert-base-multilingual-cased": "mBERT",
    "xlm-roberta-base": "XLM-R",
    "CAMeL-Lab/bert-base-arabic-camelbert-mix": "CAMeLBERT",
    "distilbert-base-uncased": "DistilBERT",
}


# Dataset location
DATA = Path("data/raw/bayan_feedback.csv")


def fertility(tokenizer, texts) -> float:
    """
    Calculate tokenizer fertility.

    Fertility = number of subword pieces / number of whitespace-separated words.

    Lower values generally indicate that the tokenizer represents
    the text using fewer subword pieces.
    """

    words = 0
    pieces = 0

    for text in texts:
        text = str(text)

        words += len(text.split())
        pieces += len(tokenizer.tokenize(text))

    return pieces / max(words, 1)


def main():
    # ---------------------------------------------------------
    # 1. Load Bayan feedback corpus
    # ---------------------------------------------------------

    if not DATA.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA}\n"
            "Make sure bayan_feedback.csv is inside data/raw/"
        )

    corpus = pd.read_csv(DATA)

    # Check required columns
    required_columns = {"text", "lang"}

    missing_columns = required_columns - set(corpus.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ---------------------------------------------------------
    # 2. Split Arabic and English text
    # ---------------------------------------------------------

    ar_texts = (
        corpus[corpus["lang"] == "ar"]["text"]
        .dropna()
        .astype(str)
        .tolist()
    )

    en_texts = (
        corpus[corpus["lang"] == "en"]["text"]
        .dropna()
        .astype(str)
        .tolist()
    )

    print(f"Arabic rows:  {len(ar_texts)}")
    print(f"English rows: {len(en_texts)}")
    print()

    # ---------------------------------------------------------
    # 3. Audit every tokenizer
    # ---------------------------------------------------------

    results = []

    for checkpoint, label in CANDIDATES.items():

        print("=" * 60)
        print(label)
        print(checkpoint)

        # Download/load tokenizer from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

        # -----------------------------------------------------
        # Fertility
        # -----------------------------------------------------

        ar_fertility = fertility(tokenizer, ar_texts)
        en_fertility = fertility(tokenizer, en_texts)

        # -----------------------------------------------------
        # Sequence lengths
        # -----------------------------------------------------

        ar_lengths = [
            len(
                tokenizer.encode(
                    text,
                    add_special_tokens=True
                )
            )
            for text in ar_texts
        ]

        en_lengths = [
            len(
                tokenizer.encode(
                    text,
                    add_special_tokens=True
                )
            )
            for text in en_texts
        ]

        # -----------------------------------------------------
        # 95th percentile
        # -----------------------------------------------------

        ar_p95 = np.percentile(ar_lengths, 95)
        en_p95 = np.percentile(en_lengths, 95)

        # -----------------------------------------------------
        # Print results
        # -----------------------------------------------------

        print(f"AR fertility: {ar_fertility:.2f}")
        print(f"EN fertility: {en_fertility:.2f}")
        print(f"AR p95 length: {ar_p95:.0f} tokens")
        print(f"EN p95 length: {en_p95:.0f} tokens")
        print()

        # Save results for later comparison
        results.append(
            {
                "model": label,
                "checkpoint": checkpoint,
                "ar_fertility": ar_fertility,
                "en_fertility": en_fertility,
                "ar_p95_length": ar_p95,
                "en_p95_length": en_p95,
            }
        )

    # ---------------------------------------------------------
    # 4. Create comparison table
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    print("=" * 60)
    print("Tokenizer Comparison")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}"
        )
    )


if __name__ == "__main__":
    main()