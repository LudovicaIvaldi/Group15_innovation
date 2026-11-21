Pipeline testing and validation with detailed explaination has been done in harm_ludo_final.ipynb.
generic_pipeline.ipynb contains the final generic pipeline, which works well in general but struggles on our sector-specific names (QUANTUM/COMPUTING/UNIVERSITY are a problem)
harmonized_patstat_valid_han.ipynb produces a validation dataset on randomized PATSTAT rows of companies with a harmonized identifier (you can select group number to increase generality of the test)
qc_applicants_PROD.csv is our prouction set, which we have to correctly cluster
harmonization_lookup.csv is the output of the generic pipeline minus the additional patstat columns (full output is in harmonized_output.csv)
specific_fixes.md is a report on the current pipeline's failure modes and possible fixes

Next step is to fuse some specific improvements from Ludovica (tf-idf notebook's work on categories for example should be added, as well as the chinese thing i do not remember)