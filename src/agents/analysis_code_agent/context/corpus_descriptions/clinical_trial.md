## CRITICAL: Corpus Structure

The corpus is sections of **clinical trial descriptions**. Each entry in `documents.jsonl` is a single clinical trial record. Documents are structured with sections including Eligibility, Conditions, Interventions, Criteria (inclusion/exclusion), Summary, and Detailed Description.

The `doc_id` format is `"{clinical_trial_id}:{section_index}"`:

- `"24073089:0"` → title/intro section of clinical trial 24073089
- `"24073089:1"` → second section 
- `"24073089:2"` → third section, etc.

All sections sharing the same prefix (e.g. `"24073089"`) belong to the **same clinical trial**. 
