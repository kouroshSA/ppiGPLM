# V3 PRS / RRS reference sets (ppiGPLM format)

The complete set of V3 (MCCV replicate) PRS/RRS **evaluation** reference sets, in
**ppiGPLM native 5-column format** (`<ps1>,SEQ1,<ps2>,SEQ2,<`, headerless). Ten
replicates (V3-1 … V3-10). Training datasets are intentionally **not** included
here.

## Contents

| folder | what it is |
|---|---|
| `PRS-RRS/` | regular PRS (positives) / RRS (random negatives), **100 pairs** each — `PRS-V3-{1..10}.csv`, `RRS-V3-{1..10}.csv` |
| `PRS-RRS_no_homodimers/` | PRS with homodimers (`SEQ1==SEQ2`) removed → **68–78 pairs**; RRS unchanged (100) |
| `PRS-RRS_homodimers_only/` | PRS with **only** the homodimers kept → **22–32 pairs**; RRS = full 100 |
| `random_controls/` | random-substituted controls: `{ps1,ps2,ps1-ps2}_random` × {PRS,RRS} × V3-{1..10}, 100 pairs each, + `manifest.csv` |

`PRS-RRS_no_homodimers/` and `PRS-RRS_homodimers_only/` are a disjoint partition
of `PRS-RRS/` (heterotypic vs. self-pair positives), both against the full RRS —
see each folder's `NOTE_RRS_homodimers.md`. The RRS never contains homodimers, so
it is the full 100-pair set in every case.

Format: `<ps1>,SEQ1,<ps2>,SEQ2,<` (trailing open marker, no label). The 2-column
ppiDCE/ppiBTEP/ppiYYD encoding of exactly the same data lives in those repos'
`V3_PRS-RRS/`. (This repo also keeps the earlier `Set_V3_PRS-RRS*` folders; this
`V3_PRS-RRS/` is the consolidated complete collection.)
