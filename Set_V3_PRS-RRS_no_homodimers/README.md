# V3 PRS / RRS — homodimer-depleted (ppiGPLM format)

Heterotypic-only versions of the V3 (MCCV replicate) reference sets in
[`../Set_V3_PRS-RRS/`](../Set_V3_PRS-RRS), with every **homodimer**
(a pair whose two sequences are identical, `SEQ1 == SEQ2`) removed from each PRS.
These are the reference sets used for the homodimer-free LES analysis of the ten
V3 models.

## Why

A homodimer is a self-interaction, detectable from the pair alone and absent
from the random reference sets, so its presence can flatter an AUC without
reflecting interaction understanding. Removing them isolates the model's ability
to discriminate *heterotypic* interactions, where the two partners differ.

(For ppiGPLM specifically, which reads a *joint* prompt rather than encoding each
partner separately, removing homodimers left discrimination essentially
unchanged — in several models it slightly *raised* AUC — indicating the models
were not resting on the self-pair shortcut.)

## Files (20)

- `PRS-V3-{1..10}.csv` — positives, homodimers removed → **68–78 pairs** each
  (down from 100).
- `RRS-V3-{1..10}.csv` — randoms, **100 pairs** each, **unchanged** (they never
  contained homodimers).

Row order is preserved; the depleted PRS is exactly the non-depleted PRS with
the `SEQ1 == SEQ2` rows dropped.

| replicate | PRS (depleted) | homodimers removed |
|---|---:|---:|
| V3-1 | 68 | 32 |
| V3-2 | 70 | 30 |
| V3-3 | 74 | 26 |
| V3-4 | 72 | 28 |
| V3-5 | 75 | 25 |
| V3-6 | 75 | 25 |
| V3-7 | 68 | 32 |
| V3-8 | 76 | 24 |
| V3-9 | 78 | 22 |
| V3-10 | 75 | 25 |

## Format

Same ppiGPLM native 5-column headerless layout as the source
(`<ps1>,SEQ1,<ps2>,SEQ2,<`); sequences and pairing are copied verbatim.
