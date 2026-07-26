# V3 PRS / RRS reference sets (ppiGPLM format)

The ten V3 (MCCV replicate) evaluation reference sets used to score the ppiGPLM
checkpoints:

- `PRS-V3-{1..10}.csv` — **positive** reference sets (reported interactions),
  100 pairs each.
- `RRS-V3-{1..10}.csv` — **random** reference sets (random non-interacting
  pairs), 100 pairs each.

These are the **non-depleted** sets (every pair kept). For the heterotypic-only
versions with self-pairs removed, see
[`../Set_V3_PRS-RRS_no_homodimers/`](../Set_V3_PRS-RRS_no_homodimers).

## Format

ppiGPLM native, 5-column, **headerless**:

```
<ps1>,SEQ1,<ps2>,SEQ2,<
```

The trailing `<` is the open marker the model scores for the interaction token;
the reference files carry no label column. (Training sets use the same layout
with the label wrapped as `<0>` / `<1>` in the fifth field.)

## Note

Each PRS carries 22–32 **homodimers** (pairs with `SEQ1 == SEQ2`); the RRS sets
carry none. Homodimers are self-interactions detectable from the pair alone, so
for a cleaner heterotypic measurement use the depleted sets linked above.
