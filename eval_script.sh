python Neural/RE/eval_spanbert.py \
  --model_dir Neural/RE/models/spanbert_nyt_re_norel \
  --proc_dir Neural/RE/processed \
  --split valid \
  --batch_size 32 \
  --max_length 256 \
  --show_per_label \
  --save_reports Neural/RE/benchmarks/spanbert_norel_valid \
  --fit_thresholds \
  --thr_grid_step 0.005 \
  --thr_min_support_pred 50

  python Neural/RE/eval_spanbert.py \
  --model_dir Neural/RE/models/spanbert_nyt_re_norel \
  --proc_dir Neural/RE/processed \
  --split test \
  --batch_size 32 \
  --max_length 256 \
  --show_per_label \
  --save_reports Neural/RE/benchmarks/spanbert_norel_test