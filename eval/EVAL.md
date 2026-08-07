# Qualitative eval log

Samples from `eval/quick_style_check.py`, appended per run -- see skills/eval/SKILL.md and PLAN.md#evaluation.

## 2026-08-03 13:49 UTC -- model: `training/output/qwen3_4b_qlora/merged`

Few-shot examples used:
- Come se metti una croce sopra al presepe: fai lo spoilerone.
- Quando l'ho visto m'è rimasto tipo tatuaggio sul cervello
- Quindi la morale è: anche se la barzelletta non fa ridere, anche se il volume non c'è, se la bella gnocca ride, la risata è contagiosa.
- Come _self non c'è nessuno.

**seed: `mare`**

> e nebbia, una stella in più, una macchia di latte… come se l'universo si fosse rotto e ti stesse cercando il cazzo.

**seed: `autobus`**

> 17, a pochi metri dalla stazione: se non la trovi non c'è.

**seed: `insonnia`**

> e merda.

**seed: `specchio`**

> specchio. E tu mi fai ridere, mi fai piangere, mi fai pensare che sono un cazzo di coglione. E se mi rido… ti rido? Tipo, io ridi e tu ridi, ridiamo insieme? Ma no. Tu… ridi da solo… io… piango. E il gatto… dorme.

**seed: `citofono`**

> : 'Come stai?' Io: 'Sto bene… Sto bene… Sto bene…' Finché ti stanco, allora mi chiami: 'Sto sull'albero… sull'albero!'


## 2026-08-03 13:51 UTC -- model: `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit`

Few-shot examples used:
- Come se metti una croce sopra al presepe: fai lo spoilerone.
- Quando l'ho visto m'è rimasto tipo tatuaggio sul cervello
- Quindi la morale è: anche se la barzelletta non fa ridere, anche se il volume non c'è, se la bella gnocca ride, la risata è contagiosa.
- Come _self non c'è nessuno.

**seed: `mare`**

> a mezza luna e un cappello di pelle

**seed: `autobus`**

> non c'è nessuno.

**seed: `insonnia`**

> di notte, pelle fredda come ghiaccio.

**seed: `specchio`**

> di una notte, con un fiore che non si apre mai.

**seed: `citofono`**

> : non risponde.

## 2026-08-05 -- Quantization benchmark

Ran `quantize/convert_and_quantize.sh training/output/qwen3_4b_qlora/merged`
(see skills/quantize/SKILL.md) to produce GGUF variants of the merged
fine-tuned model, then benchmarked each with `llama-perplexity` (held-out
`data/processed/pretrain_val.txt`, `-c 256` -- the val set is tiny, only
~881 tokens, so treat these as directional, not statistically tight) and
`llama-bench` (`-p 128 -n 64`, CPU-only, 10 threads, no GPU offload).

| Quant  | Size     | Perplexity        | Prompt proc. (t/s) | Generation (t/s) |
|--------|----------|--------------------|---------------------|-------------------|
| Q4_K_M | 2.32 GiB | 33.29 +/- 5.50     | 72.17 +/- 1.57      | 13.77 +/- 0.50    |
| Q5_K_M | 2.69 GiB | 32.19 +/- 5.26     | 19.56 +/- 1.31      | 11.75 +/- 0.40    |
| Q8_0   | 3.98 GiB | 31.51 +/- 5.15     | 23.60 +/- 1.88      | 8.85 +/- 0.47     |

**Decision: keep Q4_K_M** as the shipped quant. It's the smallest and
fastest, and the perplexity gap vs. Q8_0 (near-lossless baseline) is only
~5.6% relative -- not worth 1.7x the size or the generation-speed hit.
Q5_K_M's prompt-processing number looks anomalously low relative to Q4_K_M
and Q8_0 (non-monotonic) -- likely bench noise from the very short `-p 128`
prompt / thread contention on this machine rather than a real regression;
worth re-running with a larger prompt if this ever needs to be revisited.
The Q5_K_M/Q8_0 GGUF files were deleted after this benchmark to reclaim
disk space (regenerable any time via `quantize/convert_and_quantize.sh`).


## 2026-08-07 12:26 UTC -- model: `training/output/qwen3_4b_qlora/merged`

Generation params: temperature=0.9, top_p=0.95, repetition_penalty=1.0

Few-shot examples used:
- Come se metti una croce sopra al presepe: fai lo spoilerone.
- Quando l'ho visto m'è rimasto tipo tatuaggio sul cervello
- Quindi la morale è: anche se la barzelletta non fa ridere, anche se il volume non c'è, se la bella gnocca ride, la risata è contagiosa.
- Come _self non c'è nessuno.

**seed: `mare`**

> santo, me sento un pesce nell'acqua

**seed: `autobus`**

> 2000: era l'epoca in cui il cioccolato era dolce e la vita era dolce.

**seed: `insonnia`**

> , sconfinata, infinita, con l'occhio che sputa fumo.

**seed: `specchio`**

> . specchio.

**seed: `citofono`**

> : 'Ciao! C'ho gli occhiali da sole!'. Il volume è alto, il tono è altissimo, mi chiamo Stefano, il cazzo di cesso. Io ho un cazzo di cesso, c'entra, me lo sono costruito.


## 2026-08-07 12:27 UTC -- model: `training/output/qwen3_4b_qlora/merged`

Generation params: temperature=0.9, top_p=0.95, repetition_penalty=1.15

Few-shot examples used:
- Come se metti una croce sopra al presepe: fai lo spoilerone.
- Quando l'ho visto m'è rimasto tipo tatuaggio sul cervello
- Quindi la morale è: anche se la barzelletta non fa ridere, anche se il volume non c'è, se la bella gnocca ride, la risata è contagiosa.
- Come _self non c'è nessuno.

**seed: `mare`**

> , ciuccio, ciao. Ecco che si rifiuta di morire.

**seed: `autobus`**

> . no, autobus sull'elio.

**seed: `insonnia`**

> e stile life. E poi qua ci passa tutto: i miei problemi, le mie paure. Che a parte le paure… sono un po’ come una pila che sta per finire. Sono io, la pila e il cazzo di iPhone.

**seed: `specchio`**

> della morte.

**seed: `citofono`**

> . ma che cazzo di citofono? Il mio figlio non ha un citofono… È come se fossero le stesse cose: il vetro rotto nel giardino e i cinghie che si strappano. Ma in realtà no, sono due cose diverse! Cioè sì, per esempio io mi sento male quando vedo


## 2026-08-07 12:29 UTC -- model: `training/output/qwen3_4b_qlora/merged`

Generation params: temperature=0.7, top_p=0.95, repetition_penalty=1.0

Few-shot examples used:
- Come se metti una croce sopra al presepe: fai lo spoilerone.
- Quando l'ho visto m'è rimasto tipo tatuaggio sul cervello
- Quindi la morale è: anche se la barzelletta non fa ridere, anche se il volume non c'è, se la bella gnocca ride, la risata è contagiosa.
- Come _self non c'è nessuno.

**seed: `mare`**

> , vento, giallo. Il giallo dei gatti. Il giallo della merda. Il giallo delle zanzare. Il giallo delle barzellette.

**seed: `autobus`**

> : è il riscatto. È l'ultimo passaggio. È la fine del mondo.

**seed: `insonnia`**

> , tè, metti la birra, ridi: il tè va in tè, la birra va in birra, la ridi va in ridi. Ma il tè non va in birra. Non c'è legame. Come il cazzo e la merda. La merda è la merda. Il cazzo è il cazzo. È un cazzo di

**seed: `specchio`**

> cagionevole, lo schermo cagionevole, l'aria cagionevole.

**seed: `citofono`**

> : 'non c'è nessuno!'… ma c'è l'aria… c'è il rumore… c'è l'umore… c'è l'aria… c'è il rumore… c'è l'umore…


## 2026-08-07 12:30 UTC -- model: `training/output/qwen3_4b_qlora/merged`

Generation params: temperature=0.8, top_p=0.95, repetition_penalty=1.15

Few-shot examples used:
- Come se metti una croce sopra al presepe: fai lo spoilerone.
- Quando l'ho visto m'è rimasto tipo tatuaggio sul cervello
- Quindi la morale è: anche se la barzelletta non fa ridere, anche se il volume non c'è, se la bella gnocca ride, la risata è contagiosa.
- Come _self non c'è nessuno.

**seed: `mare`**

> sereno. o quasi. stiamo male tutti quanti. ma io sto bene perché ho le maniere di un pesce. E tu? Se sei umano e non sai come stare felice... allora smettila di soffrire e torna a essere un pesce. È più facile.

**seed: `autobus`**

> a trecento chilometri orari. Non so se esiste.

**seed: `insonnia`**

> e rabbia.

**seed: `specchio`**

> . specchio. specchio.

**seed: `citofono`**

> : 238. Ciao! La mia figlia mi chiede che cazzo stiamo facendo qua… sto cercando il culo della luna


## 2026-08-07 12:58 UTC -- model: `training/output/qwen3_4b_qlora/merged`

Generation params: temperature=0.8, top_p=0.95, repetition_penalty=1.15

Few-shot examples used:
- Come se metti una croce sopra al presepe: fai lo spoilerone.
- Quando l'ho visto m'è rimasto tipo tatuaggio sul cervello
- Quindi la morale è: anche se la barzelletta non fa ridere, anche se il volume non c'è, se la bella gnocca ride, la risata è contagiosa.
- Come _self non c'è nessuno.

**seed: `mare`**

> rosso. le mani sull'erba. un cazzo di fiore che si muove da solo. un gattino con l'uomo nel palazzo. il cesso in casa… e invece no. è il bagaglio. ma il bagaglio è più bello dei gatti. E tu? Tu sei il bagaglio. Lì dentro tutto quello

**seed: `autobus`**

> vuoto. basta un po’ di vapore e ci sono le stelle. ma tu sei sull’autostop che corre veloce e non senti niente. ti muovi a tutta birra dentro una gabbia d’acciao. Non riconosci mai le stelle perché te ne stai nella macchina.

**seed: `insonnia`**

> : come un gattino che si accorge che la sveglia sta funzionando e inizia a soffio. Poi però sputa nel bicchiere. È come quando ti stai arrampicando per vedere le nuvole e tu sei già su. Arriva quel momento dove non sai più di cosa parli, ma il vento sa tutto.

**seed: `specchio`**

> , specchio nelle tane… mi trovo nel labirinto delle tane.

**seed: `citofono`**

> che dice: "ciao, sei tu? Ciao. Si. Ah sì! Ciao, scusa, ero nella mia cabana." Tipo quando entri in un negozio e vedi tutti i prodotti che stanno a terra… e non sai bene cosa comprare.


## 2026-08-07 12:59 UTC -- model: `training/output/qwen3_4b_qlora/merged`

Generation params: temperature=0.8, top_p=0.95, repetition_penalty=1.15

Few-shot examples used:
- Come se metti una croce sopra al presepe: fai lo spoilerone.
- Quando l'ho visto m'è rimasto tipo tatuaggio sul cervello
- Quindi la morale è: anche se la barzelletta non fa ridere, anche se il volume non c'è, se la bella gnocca ride, la risata è contagiosa.
- Come _self non c'è nessuno.

**seed: `mare`**

> a cielo aperto? beh… no. il mare si chiude e torna indietro quando ti guardano nel culo.

**seed: `autobus`**

> che vanno nel vuoto. Il mondo come un autobus senza rotelle e tu sei dentro, ma stanco.

**seed: `insonnia`**

> . come un elefante che mangia pomodoro. ti sforzi e… si blocca tutto. Poi ricomincia da zero. Lo stesso schema per l'anima. Si riempie di storie. E poi viene il momento in cui le storie vanno buttate giù. Come i vestiti. Ti piaceva il blu? Il

**seed: `specchio`**

> . specchio delle merde.

**seed: `citofono`**

> ? Ciao! No. Non c'era niente. Soprattutto non era lì per te.

