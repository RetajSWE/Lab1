# Lab Notes

## Lab 1 — Defect Safari

Inspect `data/raw/bayan_raw_sample.csv` and document at least six defect classes.
For each one record: example, why it matters, and clean/preserve/task-dependent.

### Defect 1

* Class: Unicode forms
* Example: `ألطريق`, `ألإنارة`, `أحتساب`, `ألعأب الأطفال`
* Why it matters: Different Unicode forms or inconsistent Arabic character representations can cause the same or similar words to have inconsistent representations for downstream NLP models.
* Decision: Clean

### Defect 2

* Class: Tatweel
* Example: `لووووسمحت`
* Why it matters: Tatweel is mainly decorative and does not usually add semantic meaning, so it can create unnecessary variation in the text.
* Decision: Clean

### Defect 3

* Class: Code-switching Arabic ↔ English
* Example: Arabic and English feedback in the same bilingual dataset, including service names such as `Balady`, `My Licence`, and `Facilities Portal`.
* Why it matters: Bayan processes both Arabic and English, so removing or aggressively changing one language could destroy useful information.
* Decision: Preserve

### Defect 4

* Class: PII
* Example: Phone number `0551234567` and national-ID-shaped number `1023456789`
* Why it matters: Personally information should not be exposed to downstream models or services.
* Decision: Mask

### Defect 5

* Class: Emoji
* Example: `😡`
* Why it matters: Emoji can contain useful sentiment and emotional information that may help downstream NLP tasks such as sentiment analysis.
* Decision: Preserve

### Defect 6

* Class: HTML remnants
* Example: `<br>`
* Why it matters: HTML tags are formatting artifacts and do not provide useful linguistic information to the NLP pipeline.
* Decision: Clean

## Lab 2 — Parameter audit

| Checkpoint | Total params | Embeddings % | Other notes                                                                                                                 |
| ---------- | -----------: | -----------: | --------------------------------------------------------------------------------------------------------------------------- |
| mBERT      |  177,853,440 |       51.85% | 92,208,384 embedding params; 28,366,848 attention params; 56,669,184 FFN params; 18,432 norm params; 590,592 pooler params. |
| CAMeLBERT  |  109,081,344 |       21.49% | 23,436,288 embedding params; 28,366,848 attention params; 56,669,184 FFN params; 18,432 norm params; 590,592 pooler params. |

### Embedding share explanation

The embedding share is different because mBERT is multilingual and uses a much larger vocabulary, so its embedding layer contains substantially more parameters than the Arabic-focused CAMeLBERT model.

* Distribution:
* One-sentence implication for MSA-only evaluation:
