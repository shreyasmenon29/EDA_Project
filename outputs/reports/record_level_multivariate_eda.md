# Record-Level Multivariate EDA

Records analysed: **12,000**

This analysis examines relationships between document length, paragraph structure, vocabulary diversity, sentence characteristics, and legal-reference density at the individual judgment level.

## Important methodological note

Spearman correlations are used because the document features are highly non-normal and contain extreme outliers. Correlation indicates association, not causation.

## Strongest record-level associations

- `article_count_record` vs `articles_per_1000_record`: rho = 0.972, p = 0.000e+00, n = 12,000
- `word_count_record` vs `vocabulary_diversity_record`: rho = -0.908, p = 0.000e+00, n = 12,000
- `section_count_record` vs `sections_per_1000_record`: rho = 0.772, p = 0.000e+00, n = 12,000
- `word_count_record` vs `paragraph_count_record`: rho = 0.732, p = 0.000e+00, n = 12,000
- `paragraph_count_record` vs `vocabulary_diversity_record`: rho = -0.582, p = 0.000e+00, n = 12,000
- `word_count_record` vs `section_count_record`: rho = 0.579, p = 0.000e+00, n = 12,000
- `vocabulary_diversity_record` vs `section_count_record`: rho = -0.557, p = 0.000e+00, n = 12,000
- `paragraph_density_record` vs `vocabulary_diversity_record`: rho = 0.470, p = 0.000e+00, n = 12,000
- `vocabulary_diversity_record` vs `average_sentence_length_record`: rho = -0.466, p = 0.000e+00, n = 12,000
- `word_count_record` vs `average_sentence_length_record`: rho = 0.415, p = 0.000e+00, n = 12,000
- `word_count_record` vs `article_count_record`: rho = 0.395, p = 0.000e+00, n = 12,000
- `paragraph_count_record` vs `section_count_record`: rho = 0.381, p = 0.000e+00, n = 12,000
- `word_count_record` vs `paragraph_density_record`: rho = -0.373, p = 0.000e+00, n = 12,000
- `vocabulary_diversity_record` vs `article_count_record`: rho = -0.344, p = 0.000e+00, n = 12,000
- `paragraph_count_record` vs `article_count_record`: rho = 0.292, p = 9.468e-235, n = 12,000

## Interpretation cautions

- Statistical significance should not be interpreted as practical importance.
- Very large samples can produce small p-values for modest associations.
- Extreme judgments can strongly influence raw relationships, so robust statistics are preferred.
- These are observational relationships and do not establish causation.