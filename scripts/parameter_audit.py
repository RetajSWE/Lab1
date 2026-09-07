"""Lab 2: parameter accounting for mBERT and CAMeLBERT."""

from transformers import AutoModel


def audit(checkpoint: str) -> dict:
    model = AutoModel.from_pretrained(checkpoint)

    buckets = {
        "embeddings": 0,
        "attention": 0,
        "ffn": 0,
        "norms": 0,
        "pooler": 0,
        "other": 0,
    }

    for name, parameter in model.named_parameters():
        count = parameter.numel()
        name_lower = name.lower()

        if "embeddings" in name_lower:
            buckets["embeddings"] += count
        elif "attention" in name_lower:
            buckets["attention"] += count
        elif any(x in name_lower for x in ["intermediate", "output.dense"]):
            buckets["ffn"] += count
        elif "layernorm" in name_lower or ".norm" in name_lower:
            buckets["norms"] += count
        elif "pooler" in name_lower:
            buckets["pooler"] += count
        else:
            buckets["other"] += count

    total = sum(buckets.values())

    result = {
        "checkpoint": checkpoint,
        "total": total,
        **buckets,
    }

    return result


if __name__ == "__main__":
    for ckpt in [
        "bert-base-multilingual-cased",
        "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    ]:
        print(ckpt, audit(ckpt))