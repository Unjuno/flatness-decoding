# Paper Finish Skill Tree

## Level 0: Core Framing
- Goal: keep the paper about successor logit dispersion as a diagnostic/control signal, not a universal quality-improvement story.
- Required files: `paper/report_draft.md`, `paper/decoding_literature_positioning.md`, `paper/successor_dispersion_vs_likelihood.md`, `paper/writingprompts_results.md`
- Completion criteria: terminology is consistent; the method is framed as N-candidate reranking with a successor-state signal; the signal is described as likelihood-related but not identical to PPL.
- Failure signs: Shannon entropy language, orthogonality claims, or quality-superiority language appear in the draft.
- Recommended next action: review the discussion and abstract for overclaims.

## Level 1: Evidence Integrity
- Goal: ensure every claim is backed by a summary table, qualitative audit, or diagnostic note.
- Required files: `results/summary/writingprompts_experiment_summary.md`, `results/summary/comparison_gpt2_writingprompts_merged_table.csv`, `results/summary/comparison_distilgpt2_writingprompts_merged_table.csv`, `results/analysis/successor_dispersion_likelihood_diagnostics.md`, `results/analysis/writingprompts_qualitative_evidence_map.md`, `results/analysis/paper_claim_audit.md`
- Completion criteria: every important claim can be traced to at least one artifact.
- Failure signs: unsupported claims about universal quality or formal evaluation.
- Recommended next action: use the claim audit as the pre-submission checklist.

## Level 2: Quantitative Results
- Goal: make the summary tables easy to cite and interpret.
- Required files: `results/summary/writingprompts_key_results.csv`, `paper/figures/writingprompts_gpt2_flatness_vs_n.csv`, `paper/figures/writingprompts_gpt2_tradeoff_table.csv`, `paper/ppl_column_note.md`
- Completion criteria: PPL is clearly defined as the summary-table column; runtime and N-trial trade-offs are readable; baseline vs selector comparisons are explicit.
- Failure signs: ambiguity about which PPL is being reported or confusion between current-distribution and successor-state quantities.
- Recommended next action: verify figure/table captions against the source CSVs.

## Level 3: Qualitative Synthesis
- Goal: show the reader what the methods actually do in text.
- Required files: `results/summary/comparison_gpt2_writingprompts_qualitative_audit.md`, `results/summary/comparison_distilgpt2_writingprompts_qualitative_audit.md`, `results/analysis/writingprompts_gpt2_small_case_review_report.md`, `results/analysis/writingprompts_qualitative_evidence_map.md`, `results/analysis/writingprompts_qualitative_sample_bank.md`
- Completion criteria: failure modes are named consistently; minflat, maxflat, centerflat, and highprob_centerflat are distinguished clearly.
- Failure signs: the synthesis reads like a benchmark leaderboard rather than a failure-mode analysis.
- Recommended next action: mine the sample bank for a few paper-safe illustrative examples.

## Level 4: Related Work Positioning
- Goal: connect the method to decoding literature without overclaiming novelty.
- Required files: `paper/decoding_literature_positioning.md`, `paper/report_draft.md`
- Completion criteria: the paper distinguishes current-distribution decoding from candidate-induced successor-state shape; Mirostat, typical sampling, top-k/top-p, and contrastive search are all positioned correctly.
- Failure signs: the draft implies that existing methods are irrelevant or that the new signal is orthogonal to likelihood.
- Recommended next action: align the introduction and related-work language with the positioning note.

## Level 5: Writing and Figures
- Goal: keep the manuscript readable and figure/table captions cautious.
- Required files: `paper/report_draft.md`, `paper/figure_captions.md`, `paper/figures/*`
- Completion criteria: captions are cautious; figures are derived from existing CSVs; the paper draft uses the same terminology as the captions.
- Failure signs: captions imply universal quality improvement or formal human evaluation.
- Recommended next action: do a final terminology pass over abstract, discussion, and captions.

## Level 6: Reproducibility and Archiving
- Goal: document what was run, what was summarized, and how to preserve the completed artifacts.
- Required files: `docs/reproducibility_writingprompts.md`, `docs/artifact_archiving.md`, `docs/paper_finish_skill_tree.md`
- Completion criteria: archive instructions exist; raw outputs are preserved externally; the completed experiment names and prompt artifact are recorded.
- Failure signs: missing archive guidance or references to non-existent config names.
- Recommended next action: verify that README links to the reproducibility and archive notes.

## Level 7: Reviewer Defense
- Goal: prepare the paper for the most likely objections.
- Required files: `results/analysis/paper_claim_audit.md`, `results/analysis/successor_dispersion_likelihood_diagnostics.md`, `results/analysis/writingprompts_qualitative_evidence_map.md`
- Completion criteria: the draft can answer “Is this just PPL?”, “Can top-p already do this?”, and “Does it improve quality?” without overclaiming.
- Failure signs: reviewer objections are answered by stronger-than-evidence language.
- Recommended next action: use the claim audit as the final language filter before manuscript circulation.

## Final Checklist
Ready for internal manuscript review if:
- terminology is consistent;
- no Shannon entropy overclaim remains;
- PPL relation is stated as related but not identical;
- figures are generated from existing CSVs;
- qualitative analysis is labeled exploratory;
- archive instructions exist;
- raw outputs are preserved externally;
- README points to the right artifacts.
