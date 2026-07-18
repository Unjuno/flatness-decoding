# 18形状の自然言語出力評価

保存済み3モデル×18形状×20 prompt（計1080 continuation）を共通の語彙・文体ルーブリックで評価した。これは人間の意味理解評価ではなく、再現可能な表層NLP診断である。

## 評価軸

- narrative continuation: 動作・出来事・文の連鎖があるか
- prompt grounding: prompt語彙の接地があるか
- style consistency: 文章としての文長・文の連続性
- meta/list drift: タイトル・要約・箇条書き・説明への逸脱
- repetition / fragment / format breakdown: 反復・断片・記号崩れ

## 形状別の自然言語解釈

### `heavy_tail`

- narrative寄り: **gemma** (score=0.09)
- 逸脱・反復寄り: **gemma** (meta/list=0.40, repeat=0.001)
- 解釈: 反復を抑えやすいが、局所的な意味接地が弱くなり得る。
- 代表的な継続 (gemma/wp_000012): The wealthy were triumphant - the bunkers were impenetrable, shielded from the atmospheric changes, the seismic activity, and the radiation. Millions died. Billions were left to suffer. But the elite, the last of a new breed of humanity, had a new world. A world of pristine air, 
- 反例・逸脱例 (gemma/wp_000016): you are a professional bulletproof vest and carrier, ballistic armor, ballistic vest, armor, safety, ballistic armor, ballistic vest, protective, hazardous, safety, protective, personal, ballistic, safety, personal, price, cheap, best, cheap armor, armor offers, armor, armor **I 

### `high_concentration`

- narrative寄り: **gemma** (score=0.10)
- 逸脱・反復寄り: **gemma** (meta/list=0.48, repeat=0.001)
- 解釈: 流れを強く固定する。正しい流れなら流暢だが、誤ると反復・縮退する。
- 代表的な継続 (gemma/wp_000004): - Tyrion Lannister I find this saying interesting, and not just because it's from Game of Thrones, which is where I first stumbled across it. It’s got real weight to it, and reflects feelings I've experienced in life. When I was younger, I often felt like I *had* to be "good" to 
- 反例・逸脱例 (qwen/wp_000013): Fast forward several decades later . One of those appararatus druggers finds the old flag in an odometer oddity in an apparently Intergalactic ruin covered in holograms outside Lagos , Nigeria . What is a good title for this piece of writing? To determine a good title for this pi

### `high_spread`

- narrative寄り: **gemma** (score=0.12)
- 逸脱・反復寄り: **gemma** (meta/list=0.58, repeat=0.001)
- 解釈: 反復を抑えやすく、語彙の探索を許すが、物語の焦点が緩む可能性がある。
- 代表的な継続 (qwen/wp_000001): :) The car was a vintage Ford Mustang, its chrome and red paint a mere shell of the sleek, powerful muscle that had once reigned. The engine rumbled with a familiar, comforting grumble, the exhaust a steady stream of slightly burnt gasoline, the smell of freshly opened New York C
- 反例・逸脱例 (gemma/wp_000001): **Title:** Static & Rust **Word Count:** Approximately 2000 --- **Perspective 1: Silas (The Driver)** The cassette player spat out a burst of Billy Joel, the kind of nostalgic, slightly melancholic tune that matched the mood perfectly. Rain was licking at the windshield, blurring

### `large_gap`

- narrative寄り: **qwen** (score=0.06)
- 逸脱・反復寄り: **gemma** (meta/list=0.54, repeat=0.000)
- 解釈: 一つの流れへのコミットが強く、誤った流れから戻りにくい。
- 代表的な継続 (qwen/wp_000004): he replied ... " How would the previous sentence be correctly capitalized? [ WP ] "You either die a villain, or you live long enough to become the hero . " He replied, "You either die a villain, or you live long enough to become the hero. " He replied, "You either die a villain, 
- 反例・逸脱例 (gemma/wp_000001): </h1> Write a road trip story from three perspectives. The driver, the hitchhiker, and the person tied up in the trunk. **Title: Chrome Dust and Static** **Characters:** * **Miles:** (Early 30s) - Gruff, lonely driver. Haunted by a recent loss and seeking escape. * **Jasper:** (L

### `low_concentration`

- narrative寄り: **gemma** (score=0.21)
- 逸脱・反復寄り: **gemma** (meta/list=0.38, repeat=0.000)
- 解釈: 単一候補への固定が弱く、文体が安定する場合と弱くなる場合がある。
- 代表的な継続 (gemma/wp_000013): They were always strange , the two biggest arguments for the US were the one to that was a futile but constant struggle to prove that The American experiment had some meaning ( America was always to America, always never in a global if you are not a native the area or local econo
- 反例・逸脱例 (gpt2/wp_000008): -- is a delightful and humorous scene; calls upon the two (the t gtivhe ravije), the text be per se. [* He means, " lastoR the Actual Ant", by Baz.) Is this decent and frivible ?.. to help an ignorant American, to leave a diEintrable an inrap. An instantly ? If ;; srdR?Cng is ove

### `low_spread`

- narrative寄り: **qwen** (score=0.07)
- 逸脱・反復寄り: **gemma** (meta/list=0.53, repeat=0.000)
- 解釈: 候補が狭く、自然な継続にも反復・形式崩れにも固定される。モデル依存性が強い。
- 代表的な継続 (qwen/wp_000012): To the alien aliens traveled back in time , they were supposed to save humanity and once the 1984 convoy arrived aliens traveled back in time and once the 1984 convoy arrived , which corresponded the last senda car JAMDIDI sendadas era ama efeguro asusu bodofori again get an alie
- 反例・逸脱例 (gemma/wp_000000): . . ] So, that’s the prompt… Here’s what I’ve got: **Name:** Lysandra “Lys” Volkova **Age:** 28 (Appears 24 due to genetically optimized soldiers) **Rank (Pre-Ragnarok):** Sergeant - Tactical Operations **Unit:** Chronos Brigade – 34th Tactical Regiment **Equipment (Post-Ragnarok

### `mid_concentration`

- narrative寄り: **gpt2** (score=0.16)
- 逸脱・反復寄り: **gemma** (meta/list=0.55, repeat=0.000)
- 解釈: 比較的妥当な継続と探索性の妥協点に近い。
- 代表的な継続 (gpt2/wp_000012): But here's the thing. Third world nations are populated easily , peer among the buildings in your neighborhood and then suddenly you're walked out of your home by your hostel owner looking for a bomb alert-- the people the prying eyes of all these clueless, were you looking conci
- 反例・逸脱例 (gemma/wp_000001): **Title:** Static & Dust **Prompt:** Three passengers – a driver named Leo, a wide-eyed hitchhiker named Finn, and a terrified, trapped individual named Silas. Their predetermined route: to a remote, abandoned observatory. Leo is desperate to escape something, Finn is escaping so

### `mid_gap`

- narrative寄り: **gemma** (score=0.08)
- 逸脱・反復寄り: **gemma** (meta/list=0.53, repeat=0.000)
- 解釈: 候補間の選択が極端でなく、比較的安定した遷移を作る。
- 代表的な継続 (gemma/wp_000019): </h1> [ Correction: The phrase “undersuffered” should read "underwent".] The world was a mess. The sudden ability to possess, and the reciprocal reaction of “humanic possession” – where a demon was trapped, held captive, and actively *suffering* within a human host, was tearing i
- 反例・逸脱例 (gemma/wp_000002): The museum * The Grand Gallery of the Unwanted * collects and exhibits these imperfect humans . One of the prized possessions is Elias Vance . He cannot feel pain . No pleasure either . Just an empty shell . He doesn't know * why * . A curator , Silas , is fascinated by Elias and

### `mid_spread`

- narrative寄り: **gpt2** (score=0.27)
- 逸脱・反復寄り: **gemma** (meta/list=0.46, repeat=0.000)
- 解釈: 極端な逸脱を避けやすい中間状態だが、明確な意味品質の保証ではない。
- 代表的な継続 (gpt2/wp_000013): So what they raise up after hearing weñors praise Kennedy as beautiful archangels the only one who did jump up from wheelchair blind only one day in Pittsburgh when he seems to remember learning he lived on Mars so he sure knows he took ice legs half a mile down under with him at
- 反例・逸脱例 (gemma/wp_000001): </h1> Okay, let's dive in to a story about a strange road trip from three perspectives, titled “Chrome & Shadows”. --- **Chrome & Shadows** **(Part 1: Miles – The Driver)** The static of the radio had settled into a comfortable drone, mirroring the rhythm of the highway stretchin

### `mid_tail`

- narrative寄り: **gemma** (score=0.16)
- 逸脱・反復寄り: **gemma** (meta/list=0.58, repeat=0.000)
- 解釈: 探索と安定性の中間で、極端な形式逸脱は比較的少ない。
- 代表的な継続 (gemma/wp_000007): A man who made his name as a murderer with the frightening skill and dexterity demonstrated by small, regional crimes continued to operate with a chillingly obsessive dedication. But unlike many killers—particularly those whose motives were often driven by passion or personal gri
- 反例・逸脱例 (gemma/wp_000000): ** Character Creation:** **1. Character Name:** (Choose something that sounds Norse or has historical roots or just cool sounding name if you want, the name is not a prerequisite for gameplay.) **2. Squad Leader (Optional):** (If you feel it can help to designate specific strengt

### `multi_peak`

- narrative寄り: **gemma** (score=0.07)
- 逸脱・反復寄り: **gemma** (meta/list=0.58, repeat=0.000)
- 解釈: 異なる文体・形式候補が競合し、自然文とメタ／記号出力が混ざりやすい。
- 代表的な継続 (gemma/wp_000015): ]. --- The rain hammered against the corrugated iron roof of the dilapidated shack, drumming out a relentless rhythm that mirrored the throbbing in my temples. Honestly, it wasn't the rain that was getting to me; it was *them*. It had started subtly. A request for a perfectly rip
- 反例・逸脱例 (qwen/wp_000018): Based on the passage above, can you summarize the role of the person reading the Choose Your Own Adventure style life book at age 18? Yes, based on the information provided in the passage, we can summarize the role of the person reading the Choose Your Own Adventure style life bo

### `rank_slope_flat`

- narrative寄り: **gemma** (score=0.09)
- 逸脱・反復寄り: **gemma** (meta/list=0.34, repeat=0.001)
- 解釈: 順位間の確率差が緩く、語彙選択は柔軟だが意味の焦点は保証されない。
- 代表的な継続 (qwen/wp_000019): Apparently a Scottish born Stulti was caught and " possessed " by Hesn's/Thaskan'c undead immortal, Bacchnf XagaddY FX --- whose were a *second "master* which is a "Hell Uk'tyd/i' thulft'sh[ a "H' tye ll' an Its]) Lum Xad Yxhar ( a 'Here The All / The Great Arch'-s loyal +*An' ny
- 反例・逸脱例 (gemma/wp_000001): **The Rough Road** **Character Perspectives:** * **Jules (The Driver):** A weary, haunted mechanic with a penchant for solitude and a hidden secret. * **River (The Hitchhiker):** A charismatic drifter, obsessed with collecting unique experiences and a mysterious past. * ** Silas 

### `rank_slope_mid`

- narrative寄り: **qwen** (score=0.08)
- 逸脱・反復寄り: **gemma** (meta/list=0.49, repeat=0.000)
- 解釈: 柔軟性と決定性の中間で、モデルによって自然な継続に寄る。
- 代表的な継続 (gpt2/wp_000017): Dear God ! You know that " Nobles " -virtues goddesses and peoples have political and cultural impact on our culture throughout thousands of years said dynasties from Islamic period based ones in the Proto-Platonism country of Cebuistan in which Khaba'ula was born , and from over
- 反例・逸脱例 (gpt2/wp_000006): nobody sees a closer threat is always unpredictable . It brings you closer to wanting alone. It gives you friends ... Yet it does, at least the idea of what you want signals confirmation that you put yourself in the man's shoes. 11) UTILIZATION It might be wasted your way back in

### `rank_slope_steep`

- narrative寄り: **gpt2** (score=0.11)
- 逸脱・反復寄り: **gemma** (meta/list=0.69, repeat=0.000)
- 解釈: 上位候補への決定性が高く、出力が短絡的に固定される場合がある。
- 代表的な継続 (qwen/wp_000013): Fast forward several decades to 1980, the United States wins the Undersea Conquering Race back home on Earth at 1, 000 feet of below sea level salt water leading the South and North Poles close behind . Based on the passage above, answer the following question: Where could there 
- 反例・逸脱例 (qwen/wp_000015): Need I point out to you that civilization around the electromagnetic spectrum didn't pick you as god for nothing ? Based on that paragraph can we conclude that this is true about the character ? Select from: A). Yes. B). No. The thought process: To determine if we can conclude th

### `small_gap`

- narrative寄り: **gpt2** (score=0.15)
- 逸脱・反復寄り: **gemma** (meta/list=0.45, repeat=0.000)
- 解釈: 候補競合が強く、表現の揺れやモード混在が起きやすい。
- 代表的な継続 (qwen/wp_000001): :) I had the driver but I wanted to kill a moment that would make me remember for eternity Onwards : About a young man whose friend had been kidnapped and the angst that would follow. Emily , who worked in a meat joint , was sitting on the boardwalk, wishing the hell it was 4 am 
- 反例・逸脱例 (gpt2/wp_000008): -- is a change. . If, after historical materialist use, progress in human evolution approaches infinite time, time travel or just superior technological life in a theist but he is too certain of triumph to accept the depoplatory theory, then he also will seen if he will directly 

### `smooth_top`

- narrative寄り: **gemma** (score=0.18)
- 逸脱・反復寄り: **gemma** (meta/list=0.36, repeat=0.000)
- 解釈: 複数候補を残し、反復を避けやすいが、モードが弱くなる可能性がある。
- 代表的な継続 (qwen/wp_000011): The Empire is fleeing their homes and their first in command Captain "B_F" Berling is a respected operative and former terrorist , and the professional woman "V_i_g_","love the teams that can lift some planetary justice, and "W_i_n_th_i_a_r_T_e_." The week they were the group was
- 反例・逸脱例 (gpt2/wp_000008): -- isolativatezen ("The Two Step C & E by 1Bar" ~ Ann Marie Leach [ Longevity & Pretty Sunny jstby lazy life] February 2012, downloads, 29th revision (His Highest surprising number), st \ · Dato&294and disc. Wikipedia in, spoilers - via old cancerybarn's INST, dennismott and Zhzy

### `spiky_top`

- narrative寄り: **gemma** (score=0.08)
- 逸脱・反復寄り: **gemma** (meta/list=0.42, repeat=0.001)
- 解釈: 突出候補に強く依存し、流暢さと反復リスクが同居する。
- 代表的な継続 (gpt2/wp_000000): Topic] You were a soldier in a future conflict , until your heroic death . Now you face the battlefield of the Ragnarok of Nordic myth , with your full war kit Topic] You were a soldier in a future conflict , until your heroic death . Now you face the battlefield of the Ragnarok 
- 反例・逸脱例 (qwen/wp_000000): This will obliterate monsters , allowing you to clear a worldly plane of obsidian fungi . Based on the context given in the blanks, choose the most suitable option to fill in the ①. OPTIONS: [-He] [-She] [-It] [-None of the above choices] To determine the most suitable option to 

### `thin_tail`

- narrative寄り: **gpt2** (score=0.05)
- 逸脱・反復寄り: **gemma** (meta/list=0.43, repeat=0.014)
- 解釈: 候補の裾が薄く、既知の強いパターンへ戻りやすい。モデルによって反復が強くなる。
- 代表的な継続 (gpt2/wp_000000): You were planning to hop into first place here Illumination Epictetus Here's Part One Illumination Epictetus Here's Part Two Illumination Epictetus Here's Part Three Illumination Epictetus Here's Part Four Illumination Epictetus Here's Part Five Illumination Epictetus Here's Part
- 反例・逸脱例 (qwen/wp_000006): According to scientists back in 2012 ! Select your answer from the options. What most naturally follows from the above sentence ? OPT: (I). The weather must have cleared . (II). One of your friends or family may question what has happened . (III). There should have been clear sig

## 結論

形状は自然言語の意味を直接表さないが、文章が一つの物語モードに留まるか、反復・メタ説明・形式崩れへ逸脱するかという生成状態には関係する。形状の自然言語的意味はモデル依存であり、18形状名を普遍的な品質ラベルとして扱うことはできない。

## 限界

- prompt接地は語彙重複を使った近似で、意味的な設定維持ではない。
- narrative判定は動詞・文構造のヒューリスティックで、文脈理解ではない。
- 代表例は自動抽出であり、人手評価ではない。最終的な意味品質の主張には人手またはLLM評価者との一致検証が必要である。
