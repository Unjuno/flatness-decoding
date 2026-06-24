# GPT-2 WritingPrompts Small Case Review

## 1. Scope and caveat
This is a small LLM-assisted blind-ish case review of GPT-2 WritingPrompts continuations. It is exploratory, not formal human evaluation, and not a full benchmark.

## 2. Workflow
- 6 prompt IDs
- 8 conditions: top_k, top_p, random N=8, minflat N=8, maxflat N=8, centerflat_q50 N=8, centerflat_q75 N=8, highprob_centerflat N=8
- Anonymized scoring was completed before joining the hidden key
- Seed 0 was used when available; in this slice, all selected items were available at seed 0

## 3. Main observed failure modes
The most common failure modes were prompt drift, list-like or metadata-style text, generic filler, and repetition/looping. The worst cases also showed symbol or encoding noise, URL or markup artifacts, and code-like fragments.
After unblinding, these issues were concentrated in `minflat` and the `highprob_centerflat` / baseline extremes more than in the center-flat selectors, although the center-flat outputs still often failed to stay on prompt.

## 4. Method-level observations
- `top_k` / `top_p`: mostly fluent at the sentence level, but they frequently drift into wiki-like, essay-like, or instructional text instead of sustaining the prompt.
- `random` N=8: somewhat less collapsed than the worst extremes, but still often off-prompt and generic.
- `minflat` N=8: the clearest failure mode. It tends to produce the most unusable outputs, including symbol noise, encoding drift, and repetition collapse.
- `maxflat` N=8: less catastrophic than `minflat`, but still frequently diffuse, generic, or metadata-like.
- `centerflat_q50` / `centerflat_q75` N=8: these are the most plausible center-region outputs in the sample, but they are still often only partially usable and do not consistently solve prompt drift.
- `highprob_centerflat` N=8: conservative and relatively stable at the token-choice level, but it can become repetitive, markup-like, or overly generic.

## 5. Evidence examples
- `top_k`: `Can you save your village and live well in that city or will you be slaughtered by the god`
- `top_p`: `Can you show your bravery and honour well in battle? — Aaeged Below Police 16 Tirmidaren t`
- `random`: `Loaded with their inhuman weaponry - smooth and rigid. Now equipped with your new upgraded`
- `minflat`: `   ⏕🏆 ⏕⏕Є⏄léer 🏆5%⏃ ⏙ ⏗ ⏖ ⏖⏃ ⏄з ⏗⏖ ⏗☌ ☑ ⏗☽ ☐☑ ⏗☐ ☐ В⏗ ☎ ⏗ К⏗ ☴ ⏗ Т☴ ☴ ☮ ☬ ж ⏗☄ ⏗☄ ⏗☄☠☠ `
- `maxflat`: `Please don't stand by and passively passively accepting.  Remarkable  Holidays are sometim`
- `centerflat_q50`: `Remember that great shield of yours in the image of King Ainz Ooal Gown?  ] You were a sol`
- `centerflat_q75`: `<br /><br /> This weapon is the best of the best for mounting a Pyre at his side and also `
- `highprob_centerflat`: `<br /> [ WP ] This is a really simple game, but with a very unique atmosphere. Everyone he`

## 6. Relation to quantitative results
The case review is consistent with the quantitative tables in broad outline: `minflat` is the strongest dispersion extreme and is also the least usable; `maxflat` is weaker than the center-flat selectors; and `centerflat_q50` / `centerflat_q75` remain the most promising compromise settings. The higher-likelihood control (`highprob_centerflat`) tends to look safer than the extremes, but the review shows that safety often comes with repetition or genericness rather than clear story quality.
The key-results table and merged tables also suggest that N-trial control changes runtime materially, so the qualitative evidence here supports treating `N=8` or `N=16` as the practical knee unless a larger `N` is needed for a specific experimental argument.

## 7. Safe paper-language paragraph
This small blind-ish case review of GPT-2 WritingPrompts outputs suggests that successor logit dispersion is a controllable decoding-time signal, but the failure modes are not eliminated by moving toward the center region. The center-flat selectors appear more balanced than the extremes, while `minflat` is the clearest diagnostic extreme and `highprob_centerflat` is the most conservative setting. These observations are exploratory and do not constitute formal human evaluation.
