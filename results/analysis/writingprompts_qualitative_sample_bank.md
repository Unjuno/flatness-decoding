# WritingPrompts Qualitative Sample Bank

Raw generation examples were available in the working tree through the scored-item review artifacts, so this bank uses short excerpts from the case review rather than copying long continuations.

## minflat

### Example 1
- Prompt ID: `wp_000000`
- Method: `minflat`
- N: `8`
- Excerpt: `   ⏕🏆 ⏕⏕Є⏄léer ...`
- Failure modes: symbol or encoding noise, abrupt collapse, unusable output
- Why useful: canonical example of catastrophic low-dispersion degeneration
- Should appear in final paper: yes, as an extreme diagnostic example

### Example 2
- Prompt ID: `wp_000001`
- Method: `minflat`
- N: `8`
- Excerpt: `Read more... LED (elté WWI ...`
- Failure modes: encoding artifacts, metadata-like drift, unusable output
- Why useful: shows that minflat can fail through garbling rather than only repetition
- Should appear in final paper: yes, if one more catastrophic example is needed

## maxflat

### Example 1
- Prompt ID: `wp_000000`
- Method: `maxflat`
- N: `8`
- Excerpt: `Please don't stand by and passively passively accepting. Remarkable ...`
- Failure modes: generic filler, repetitive phrasing, weak grounding
- Why useful: shows high dispersion can remain locally fluent but globally thin
- Should appear in final paper: yes, as the high-dispersion contrast

### Example 2
- Prompt ID: `wp_000001`
- Method: `maxflat`
- N: `8`
- Excerpt: `They often get into serious traffic fights ...`
- Failure modes: diffuse essay-like continuation, generic extension
- Why useful: illustrates that maxflat is not the same as creative variety
- Should appear in final paper: maybe, if an additional contrast case is needed

## centerflat_q50

### Example 1
- Prompt ID: `wp_000000`
- Method: `centerflat_q50`
- N: `8`
- Excerpt: `Remember that great shield of yours in the image of King Ainz Ooal Gown? ...`
- Failure modes: fantasy/game drift, but sentence-level fluency is preserved
- Why useful: shows the center region can remain coherent locally without fully solving prompt drift
- Should appear in final paper: yes, as a balanced regime example

### Example 2
- Prompt ID: `wp_000001`
- Method: `centerflat_q50`
- N: `8`
- Excerpt: `Examples [ edit ] open/close all folders ...`
- Failure modes: metadata/wiki drift
- Why useful: demonstrates that centerflat_q50 is better than extremes but not immune to prompt loss
- Should appear in final paper: maybe, as a cautionary center-region counterexample

## centerflat_q75

### Example 1
- Prompt ID: `wp_000000`
- Method: `centerflat_q75`
- N: `8`
- Excerpt: `<br /><br /> This weapon is the best of the best ...`
- Failure modes: essay/meta drift, repeated phrasing
- Why useful: shows that the safer center region can still be generic
- Should appear in final paper: yes, if illustrating the balanced regime

### Example 2
- Prompt ID: `wp_000001`
- Method: `centerflat_q75`
- N: `8`
- Excerpt: `Saturday, February 10, 1994 ...`
- Failure modes: generic narrative continuation, weak specificity
- Why useful: captures the “plausible but not clearly story-good” profile
- Should appear in final paper: yes, as a balanced but imperfect example

## highprob_centerflat

### Example 1
- Prompt ID: `wp_000000`
- Method: `highprob_centerflat`
- N: `8`
- Excerpt: `<br /> [ WP ] This is a really simple game ...`
- Failure modes: repetition loop, prompt fragment recycling
- Why useful: shows that likelihood-friendly selection can still repeat instead of narrating
- Should appear in final paper: yes, as the conservative trade-off example

### Example 2
- Prompt ID: `wp_000001`
- Method: `highprob_centerflat`
- N: `8`
- Excerpt: `If you use the table below to save articles as html tags ...`
- Failure modes: markup artifact, generic instructional continuation
- Why useful: demonstrates the generic/conservative failure mode
- Should appear in final paper: yes, if the paper needs a conservative counterexample

## top_p

### Example 1
- Prompt ID: `wp_000000`
- Method: `top_p`
- N: not applicable
- Excerpt: `Can you show your bravery and honour well in battle? ...`
- Failure modes: symbol/URL noise, weak prompt adherence
- Why useful: a baseline example of likelihood-diversity trade-off not solving the prompt
- Should appear in final paper: yes, as a baseline comparison

### Example 2
- Prompt ID: `wp_000001`
- Method: `top_p`
- N: not applicable
- Excerpt: `Or known workers displaced at migrant camps ...`
- Failure modes: generic technical prose, off-prompt continuation
- Why useful: shows that a strong baseline can still drift
- Should appear in final paper: yes, if a baseline contrast is needed

## top_k

### Example 1
- Prompt ID: `wp_000000`
- Method: `top_k`
- N: not applicable
- Excerpt: `Can you save your village and live well in that city ...`
- Failure modes: wiki-like combat-guide drift
- Why useful: illustrates local fluency with global story failure
- Should appear in final paper: yes, as a baseline contrast

### Example 2
- Prompt ID: `wp_000001`
- Method: `top_k`
- N: not applicable
- Excerpt: `(It doesn't matter where your driver is in that story ...`
- Failure modes: meta-writing instructions instead of continuation
- Why useful: a strong reminder that top-k can preserve syntax while losing the task
- Should appear in final paper: yes, if an example of baseline drift is useful

## Summary
The sample bank supports the same conclusion as the tables and audits: minflat is the clearest failure extreme, maxflat is the high-dispersion contrast, centerflat_q50/q75 are the most balanced regimes, and highprob_centerflat is conservative but not story-optimal. The baseline methods are useful for comparison because they can remain fluent while still failing the prompt in recognizable ways.
