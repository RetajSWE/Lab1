# Decision Records

## tokenizer
## tokenizer
- Chosen checkpoint(s): XLM-R (`xlm-roberta-base`)
- Arabic fertility evidence: 1.67
- English fertility evidence: 1.43
- p95 length evidence: AR = 21 tokens, EN = 23 tokens
- Operational trade-off / rationale: XLM-R provides the most balanced bilingual performance for Bayan. It has low Arabic fertility and the lowest English fertility among the multilingual candidates, with relatively short p95 sequence lengths for both Arabic and English. CAMeLBERT has slightly better Arabic fertility (1.41 vs. 1.67) but substantially worse English fertility (2.70) and English p95 length (38 tokens). DistilBERT has poor Arabic performance, while mBERT has higher fertility and longer sequences than XLM-R.

## arabic-model
- Incumbent:
- Candidate:
- All/Gulf/MSA evidence:
- CI-backed verdict:
- Segmentation contract:

## search-min-score
- Threshold:
- No-answer evidence:
- False-positive / false-negative trade-off:

## quantisation-split
- Topic artefact:
- NER artefact:
- Latency evidence:
- Paired quality-tax evidence:
- Rollback artefact retained:

## architecture
- Encoder/decoder rationale by task:
- Multilingual vs Arabic-centric rationale:
- Evidence used:
