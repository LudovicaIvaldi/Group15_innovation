# Step 2 – Applicant name harmonisation

## `harmonisation_gold.ipynb`

```
names → clean → brand key → fuzzy → group_id, group_name → precision / recall / F1
```

General pipeline, tested on the course gold standard. The brand key is the first word of the cleaned name that is not a generic word (institutions, places, legal forms); fuzzy matching merges almost identical keys. Result: precision 0.96, recall 0.94, F1 0.95.

## `harmonisation_quantum.ipynb`

```
[new]    preparation: raw name = han_name, else psn_name
[new]    translation of Chinese / Japanese brand names
[new]    applicant type: keep company / university / gov, run separately for each
[reused] clean → brand key → fuzzy   (same functions, generic words + sector words)
[new]    manual fixes for the largest firms (IBM, Origin, NTT, D-Wave, Red Hat)
[new]    merge with the full qc file → harmonization_output.csv (input of the analysis)
```

Same pipeline as the gold notebook, with sector steps around it. Output: `harm_id_final` and `harmonized_name_final` for every organisation; individuals are kept with empty names.
