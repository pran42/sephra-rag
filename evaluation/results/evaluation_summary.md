# Sephra retrieval evaluation (auto-generated)

Ranked queries with rubric labels: 16. K = 5.

| System | P@5 | strict P@5* | MRR | nDCG@5 | capped R@50 | special cases | filter violations | dup. families | misattributed reviews | critical shown / available | mean latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_original | 0.650 | 0.620 | 0.771 | 0.607 | n/a | 0/3 | 33 | 4 | 0 | 4/17 | 0.122 |
| dense | 1.000 | 0.840 | 1.000 | 0.953 | 0.903 | 3/3 | 0 | 0 | 0 | 18/18 | 0.245 |
| hybrid | 1.000 | 0.860 | 1.000 | 0.960 | 0.927 | 3/3 | 0 | 0 | 0 | 18/18 | 0.119 |
| hybrid_rerank_minilm | 1.000 | 0.820 | 1.000 | 0.943 | 0.934 | 3/3 | 0 | 0 | 0 | 17/17 | 1.787 |
| hybrid_rerank_bge-base | 1.000 | 0.840 | 1.000 | 0.961 | 0.934 | 3/3 | 0 | 0 | 0 | 20/20 | 9.717 |

*strict P@5 counts only grade-2 (highly relevant) products, averaged over the queries whose rubric has a grade-2 level.

## Per-query nDCG@5

| Query | baseline_original | dense | hybrid | hybrid_rerank_minilm | hybrid_rerank_bge-base |
|---|---|---|---|---|---|
| Q01 | 0.85 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q02 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q03 | 0.36 | 0.91 | 1.00 | 1.00 | 0.90 |
| Q04 | 0.70 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q05 | 0.85 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q06 | 0.70 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q07 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q08 | 0.60 | 0.98 | 0.97 | 0.97 | 1.00 |
| Q09 | WRONG | correct | correct | correct | correct |
| Q10 | WRONG | correct | correct | correct | correct |
| Q11 | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q12 | 0.66 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q13 | (ok) | (ok) | (ok) | (ok) | (ok) |
| Q14 | WRONG | correct | correct | correct | correct |
| Q15 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q16 | 0.55 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q17 | 0.35 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q18 | 0.86 | 1.00 | 1.00 | 1.00 | 1.00 |
| Q19 | 0.90 | 0.91 | 0.91 | 0.70 | 0.91 |
| Q20 | 0.33 | 0.45 | 0.48 | 0.42 | 0.56 |
