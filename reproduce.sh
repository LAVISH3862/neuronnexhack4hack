#!/usr/bin/env bash
# Reproduces Team Quadruples' Round 2 submissions end to end, in the order and with the settings they were made.
#   sub9  -> submissions/cand_final_resnet.csv   (public 0.9014)
#   sub10 -> submissions/cand_final_all.csv      (safe final)
#   sub11 -> submissions/cand_final_fixmatch.csv (public 0.9048, our best upload)
# Needs the competition files in ./signal-hunters-round2/, Python 3.11+, a CUDA GPU (4 GB is enough) and
# `pip install -r requirements.txt`. Total runtime on an RTX 3050 laptop GPU is roughly 12-14 hours.
# GPU training uses cudnn.benchmark, so retrained models can differ very slightly from ours.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p preds models submissions figures
export KMP_DUPLICATE_LIB_OK=TRUE
step() { echo; echo "=== $* ($(date +%H:%M))"; }

step "1. Normalised caches, panel levels, physics features"
python prep_cnn.py                                     # cache_z_{train,test}.npy, cache_valid_{train,test}.npy
python -c "import numpy as np; from common import load; from features_level import levels
[np.save(f'cache_levels_{s}.npy', levels(load(s)[0])) for s in ['train', 'test']]"
python features.py                                     # features_{train,test}.pkl
python features_extra.py                               # scintillation + line-connection features
python features_level.py                               # ON/OFF power levels (training levels get test-like gain jumps)
python make_deg.py                                     # test-like noise-degraded training copy (seed 2026)
for p in 0 1 2; do python features_deg.py $p 3; done
python -c "import pandas as pd; pd.concat([pd.read_pickle(f'features_deg_train_p{p}.pkl') for p in range(3)], ignore_index=True).to_pickle('features_deg_train.pkl')"
python features_fft.py                                 # 2D periodogram features (clean, degraded, test)
python train_gbm.py                                    # LightGBM on the features (preds/gbm_*; used by stack.py's report)

step "2. CNN v1 (5 folds) and the first stacks / pseudo-labels"
for f in 0 1 2 3 4; do python train_cnn.py --fold $f; done                    # tag cnn
python stack.py cnn                                                         # preds/stack_cnn_test.npy
LEVEL=0 python stack_pl.py cnn stack_cnn 0.9 stackpl_cnn 1                  # sub3 model (pseudo-labelled stacker)

step "3. Pseudo-labelled CNN (5 folds) and its stack"
for f in 0 1 2 3 4; do python train_cnn2.py --fold $f --no-rank --pseudo preds/stackpl_cnn_test.npy --pl-thr 0.9 --tag cnnpl; done
LEVEL=1 python stack_pl.py cnn+cnnpl stackpl_cnn 0.9 stackpl_cnn_cnnpl 3    # sub4 model

step "4. CNN v3 (zoom augmentation, class-balanced pseudo-labels)"
for f in 0 1 2 3 4; do python train_cnn3.py --fold $f --no-rank --zoom-p 0.5 --pseudo preds/stackpl_cnn_cnnpl_test.npy --pl-thr 0.9 --tag cnn3pl; done
python predict_deg.py cnn cnnpl cnn3pl                                      # CNN predictions on the test-like copy
FFT=0 MODES=C python stack_deg.py cnn+cnnpl stackpl_cnn 0.9 stackdeg3 3      # sub5 model (stacker on clean + test-like data)
MODES=C python stack_deg.py cnn+cnnpl+cnn3pl stackdeg3_C 0.9 stackdeg6 3     # sub6 model

step "5. Test-noise CNN (cnn4s1) and its stack"
for f in 0 1 2 3 4; do python train_cnn3.py --fold $f --no-rank --pseudo preds/stackdeg6_C_test.npy --pl-thr 0.9 --noise-lo 0.3 --noise-hi 1.2 --seed 1 --tag cnn4s1; done
python predict_deg.py cnn4s1
MODES=C python stack_deg.py cnn+cnnpl+cnn3pl+cnn4s1 stackdeg6_C 0.9 stackdeg_v4s1 3   # sub7 model
python blend.py submissions/cand_after_cnn4s1.csv stackdeg_v4s1_C:1

step "6. ImageNet-pretrained ResNet18 (5 folds) -> sub9"
for f in 0 1 2 3 4; do python train_pre.py --fold $f --no-rank --pseudo preds/stackdeg6_C_test.npy --pl-thr 0.9 --noise-lo 0.3 --noise-hi 1.2 --seed 1 --level --lr 6e-4 --bs 64 --tag pre; done
python blend.py submissions/sub8_recon.csv stackdeg_v4s1_C:1 pre_f0_s0:1   # sub8 (its probabilities define the noise-only test pool below)
python -c "import numpy as np; from common import LAB2ID
P = np.load('submissions/sub8_recon_probs.npy'); k = LAB2ID['noise']
np.save('cache_pks_noise_idx.npy', np.where((P.argmax(1) == k) & (P[:, k] > 0.98))[0])"
MODES=C python stack_deg.py cnn+cnnpl+cnn3pl+cnn4s1+pre stackdeg6_C 0.9 stackdeg_pre 3
python eval_final.py submissions/cand_final_resnet.csv pre                  # sub9; its probabilities are the round-2 pseudo-labels

step "7. EfficientNet-B0 (5 folds) with real test-telescope noise mixing"
PL2="--pseudo submissions/cand_final_resnet_probs.npy --pl-thr 0.9 --noise-lo 0.3 --noise-hi 1.2 --level --lr 6e-4 --pks-mix 0.5 --mixup 0"
for f in 0 1 2 3 4; do python train_pre.py --fold $f $PL2 --no-rank --arch efficientnet_b0 --bs 32 --seed 14 --tag preeff; done

step "8. FixMatch / Mean Teacher fine-tune of the ResNet folds (students saved as prefmstu)"
for f in 0 1 2 3 4; do python train_pre.py --fold $f --init 'models/pre_f{fold}_s0.pt' --fixmatch 1 --fm-tau 0.95 --fm-mu 1 --ema 0.999 \
  --no-rank --level --noise-lo 0.3 --noise-hi 1.2 --pks-mix 0.5 --lr 2e-4 --epochs 8 --bs 48 --seed 21 --tag prefm; done

step "9. Final stacks"
MODES=C python stack_deg.py cnn+cnnpl+cnn3pl+cnn4s1+pre+preeff stackdeg6_C 0.9 stackdeg_final 3
STACK_NEW=stackdeg_final STACK_OLD=stackdeg_pre python eval_final.py submissions/cand_final_all.csv pre preeff                 # sub10
MODES=C python stack_deg.py cnn+cnnpl+cnn3pl+cnn4s1+pre+preeff+prefmstu stackdeg6_C 0.9 stackdeg_fm 3
STACK_NEW=stackdeg_fm STACK_OLD=stackdeg_final python eval_final.py submissions/cand_final_fixmatch.csv pre preeff prefmstu   # sub11

step "10. Class-prior correction for the test telescope's label mix -> sub12"
python prior_correct.py stackdeg_fm submissions/cand_final_prior.csv                                                      # sub12

step "11. Same stack with cross-fitted pseudo-labels (no test row predicted by a model trained on its own pseudo-label) -> sub13"
PLCV=1 MODES=C python stack_deg.py cnn+cnnpl+cnn3pl+cnn4s1+pre+preeff+prefmstu stackdeg6_C 0.9 stackdeg_fmcv 3
python prior_correct.py stackdeg_fmcv submissions/cand_final_cv_prior.csv                                                 # sub13
python blend.py submissions/cand_final_cv.csv stackdeg_fmcv_C:1                                                           # sub14 (no prior correction)
step "done: submissions/cand_final_cv.csv (sub14), cand_final_cv_prior.csv (sub13), cand_final_prior.csv (sub12), cand_final_fixmatch.csv (sub11)"
