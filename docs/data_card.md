# Data card

| | |
|---|---|
| **Source** | [USAspending.gov](https://www.usaspending.gov) API, `spending_by_award`, contracts (types A, B, C, D) awarded by the US Department of Energy |
| **Licence** | US government data, public domain |
| **Years** | US fiscal years 2018 to 2025 (October 2017 to September 2025), up to 8,000 contracts per year |
| **Input** | `Description`: a short free-text description written by the buyer |
| **Label** | `PSC` (Product and Service Code), grouped into 14 spend categories in `spendcoder/taxonomy.py` |
| **Personal data** | none used: recipient names are not downloaded |
| **Rebuild** | `python scripts/prepare_data.py` (about 10 minutes) or step 1 of the notebook |

## Cleaning
- "IGF::...::IGF" administrative markers removed, spaces normalised, casing kept
- lines shorter than 4 characters, or without a code, dropped
- a contract that appears in several years counts once (earliest start date)

## Split
| Split | Contracts starting | Note |
|---|---|---|
| Train | before 1 Oct 2023 | |
| Validation | 1 Oct 2023 to 30 Sep 2024 | chooses the best epoch and the review threshold |
| Test | from 1 Oct 2024 | used once; lines whose exact text is in training are removed |

## Limits
- One public-sector buyer's wording and taxonomy. Another company's lines (supplier names, local language,
  internal codes) will differ: check on its own labelled lines and fine-tune further.
- The PSC is chosen by the buyer and is sometimes debatable, so part of the "errors" are label noise.
- Categories are imbalanced; macro F1 is reported so small categories count.
