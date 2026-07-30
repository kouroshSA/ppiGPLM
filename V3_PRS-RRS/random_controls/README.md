# V2 PRS / RRS — random-sequence leakage controls

**Date:** 2026-07-20
**Folder:** `OOF_Sets_V2/V2_PRS_RRS_random_controls/`
**Generator:** `../build_v2_prsrrs_random_controls.py`
**Manifest:** `manifest.csv`

> **Retagged 2026-07-29:** the 60 control filenames (and `manifest.csv`) were
> renamed `-V2-{k}-` → `-V3-{k}-` so they line up with the V3 replicates in this
> clean dataset folder. Contents are byte-identical — `V2-k` corresponds 1:1 to
> `V3-k` (verified against the ppiDCE/ppiBTEP V3 controls). The **source,
> generator, and seed history below still refer to the original V2 provenance**,
> which is unchanged.

Random-sequence control variants of the V2 PRS/RRS evaluation sets. In each
variant one or both protein sequences of every row are replaced by a **random
sequence of the same length** drawn uniformly from the 20 standard amino acids;
everything else in the row is preserved. These are negative/leakage controls:
a model relying on real sequence content should separate the intact PRS/RRS far
better than these randomized versions, and the both-randomized variant should
collapse toward chance.

## Source

`../V2_PRS_RRS_depleted/{PRS,RRS}-V2-{1..10}.csv` — bare prompts, 100 rows each:

```
<ps1>,SEQ1,<ps2>,SEQ2,<          SEQ1 = col 1 (ps1),  SEQ2 = col 3 (ps2)
```

## Variants (per source file)

| Suffix | SEQ1 (ps1) | SEQ2 (ps2) |
|---|---|---|
| `_ps1_random.csv`     | randomized | original |
| `_ps2_random.csv`     | original   | randomized |
| `_ps1-ps2_random.csv` | randomized | randomized |

Randomization uses only `ACDEFGHIKLMNPQRSTVWY`, same length as the original, i.i.d.
uniform. The `<ps1>`/`<ps2>` tags and the trailing `<` are preserved; the
non-randomized sequence is byte-identical to the source.

## Files

20 source files (PRS/RRS × versions 1–10) × 3 variants = **60 variant files**,
plus `manifest.csv`. Naming: `{PRS|RRS}-V3-{k}_{variant}.csv` (retagged from `-V2-`).

## Seeding

Every generated sequence has a globally unique, reproducible seed:

```
seed = 300_000_000
     + variant  * 10_000_000    # ps1_random=0, ps2_random=1, ps1-ps2_random=2
     + typecode *  1_000_000     # PRS=0, RRS=1
     + version  *     10_000     # 1..10
     + position *      1_000     # ps1(col1)=0, ps2(col3)=1
     + row_index                 # 0..99
```

8,000 distinct seeds across all randomized sequences, verified with 0 collisions.
This range (≥300,000,000) is disjoint from the seed ranges used by the other
randomization scripts in this project.

## Verification (all 60 files, 0 problems)

Independent re-read confirmed, per variant row:
1. `<ps1>`/`<ps2>` tags and the trailing `<` unchanged.
2. Each randomized sequence matches the original's length and uses only standard
   amino acids.
3. Each non-randomized sequence is byte-identical to the source.
4. Row counts preserved (100 each).
5. All files use **LF** line endings (0 files with CR).
6. Formula seeds unique (8,000 seeds, 0 duplicates).

## Manifest (`manifest.csv`)

Columns: `file, type, version, variant, rows, seq1_randomized, seq3_randomized,
seed_base` — one row per variant file (60 rows).

## Reproduce

```bash
cd OOF_Sets_V2
python build_v2_prsrrs_random_controls.py   # deterministic; writes this folder + manifest.csv
```

## Intended use (LES-wrapper)

Pair matching modifications only — e.g. run `PRS-V3-k_ps2_random.csv` with
`RRS-V3-k_ps2_random.csv` — so a single LES run never mixes ps1 and ps2
modifications. Each is evaluated against the checkpoints trained on that version's
depleted training set, exactly as with the native PRS/RRS sets.
