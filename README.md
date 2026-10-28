# Signal Hunters (Hack4Hack 2026, Round 2): Team Quadruples

Classifies radio-telescope ON/OFF cadences into 9 classes (ETI candidate, FRB, pulsar, scintillating star,
satellite pass, radar chirp, intermittent and persistent RFI, noise). Training data comes from the GBT telescope,
the test data from PKS, so most of the work goes into surviving that telescope shift.

**Final uploads:** `sub11` (public macro-F1 0.9048), `sub12` (sub11 with a class-prior correction, 0.9040), `sub13`
(sub12 with cross-fitted pseudo-labels in the stacker), `sub14` (sub11 with cross-fitted pseudo-labels, 0.9085), `sub15`
(cross-fitted stack with all model families, averaged over 3 test splits) `sub16` (sub15 re-run with pseudo-labels from
sub15), `sub17` (sub16 without the two models that echo their teachers) and `sub18` (sub17 with pseudo-labels from sub17), reproduced by `reproduce.sh` as `submissions/cand_final_fixmatch.csv`, `cand_final_prior.csv`,
`cand_final_cv_prior.csv`, `cand_final_cv.csv`, `cand_final_fm2cv.csv`, `cand_final_fm2cvt.csv`, `cand_final_noecho.csv`
and `cand_final_noecho2t.csv`.

## Approach

1. **Telescope-independent normalisation** (`common.py`). Dropped rows (value 0) are treated as missing. The bandpass
   shape is estimated from the OFF panels (median, then a 15-channel running median so narrow lines survive) and each
   panel keeps its own level, so ON-only emission is preserved. GBT's centre-channel spike is removed, and each panel
   is scaled by its robust noise level.
2. **Physics features** (`features.py`, `features_extra.py`, `features_level.py`, `features_fft.py`):
   - drift-line search per panel (27 drift rates) and a **track search in global time** `t = p·(16+slew_gap)+r`
     (the best straight track through the three ON panels and the OFF power along it)
   - line-connection checks against the ETI traps (lines that do not connect, or do not drift)
   - dedispersion search (undispersed glitches vs dispersed FRB/pulsar sweeps, pulse counts)
   - scintillation: broad spectral bumps in ON vs OFF, and **broadband ON/OFF power levels**
   - 2D periodogram (radar)
3. **CNNs on a "tall" cadence image**: the six panels are placed at their global time rows (slew gaps left empty) with
   an ON/OFF indicator channel and, for the pretrained models, a panel power-level channel, so an ETI track is one
   straight line through the image.
   - small CNNs trained from scratch (`train_cnn.py`, `train_cnn2.py`, `train_cnn3.py`)
   - **ImageNet-pretrained ResNet18 and EfficientNet-B0** fine-tuned on the image, upsampled 2× (`train_pre.py`)
4. **Domain adaptation to the test telescope**
   - **pseudo-labelling**: confident test predictions of the previous stage are added to training (worth about +5
     macro-F1 points on the test telescope in our ablation)
   - **test-like training data**: a noise-degraded copy of the training set matching PKS's lower signal-to-noise
     (`make_deg.py`), and training cadences mixed with **real noise-only PKS cadences** (`--pks-mix`)
   - **FixMatch / Mean Teacher** (`train_pre.py --fixmatch`): an EMA teacher re-labels test cadences online and the
     student must agree on strongly augmented views; we use the student models
5. **Stacking** (`stack_deg.py`, `eval_final.py`): a LightGBM stacker on the physics features plus out-of-fold CNN
   probabilities, trained on clean **and** test-like copies of the training data, then blended with the pretrained
   models. Blend weights are chosen by cross-fitting on out-of-fold predictions.
6. **Class-prior correction** (`prior_correct.py`, sub12): the test telescope's class mix is estimated without labels
   by inverting the confusion matrix of the test-like out-of-fold predictions, and each class's probability is
   reweighted by its estimated test share over its training share.
7. **Cross-fitted pseudo-labels** (`stack_deg.py` with `PLCV=1`, sub13): each fifth of the test set is predicted by a
   stacker trained on the other four fifths' pseudo-labels, so no test cadence is predicted by a model that saw its own
   pseudo-label.

All model selection used 5-fold out-of-fold validation on two proxies of the test telescope (the noise-degraded copy
and training cadences mixed with real PKS noise), not the public leaderboard.

## Reproduce

Put the competition files in `signal-hunters-round2/`, then:

```bash
pip install -r requirements.txt
bash reproduce.sh
```

It runs every stage in the order and with the settings used during the round (about 12–14 hours on an RTX 3050
laptop GPU) and writes `submissions/cand_final_cv_prior.csv` (sub13), `submissions/cand_final_prior.csv` (sub12),
`submissions/cand_final_fixmatch.csv` (sub11) and `submissions/cand_final_all.csv` (sub10).
GPU training uses `cudnn.benchmark`, so retrained models can differ very slightly from ours. Trained weights and
intermediate predictions are available on request.

## Files

| Stage | Files |
|---|---|
| Data, normalisation, caches | `common.py`, `prep_cnn.py`, `make_deg.py` |
| Physics features | `features.py`, `features_extra.py`, `features_level.py`, `features_fft.py`, `features_deg.py` |
| Models | `train_gbm.py`, `train_cnn.py`, `train_cnn2.py`, `train_cnn3.py`, `train_pre.py` |
| Stacking and finals | `stack.py`, `stack_pl.py`, `stack_deg.py`, `predict_deg.py`, `eval_final.py`, `blend.py`, `prior_correct.py`, `final_avg.py` |
