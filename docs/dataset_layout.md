# ACDC Layout

```text
data/acdc/
  rgb_anon/
    train/
      fog/
      night/
      rain/
      snow/
    val/
      fog/
      night/
      rain/
      snow/
  gt/
    train/
      fog/
      night/
      rain/
      snow/
    val/
      fog/
      night/
      rain/
      snow/
```

RGB files must end in `_rgb_anon.png`. Label files must end in `_gt_labelTrainIds.png`.

Normal-condition images used for synthetic restoration training may be stored in any nested image directory. Set that directory as `data.clean_root` in `configs/restoration.yaml` and `configs/task_aware.yaml`. When the ACDC reference images are stored alongside adverse images under `rgb_anon/<condition>/<split>_ref`, set `clean_root: data/acdc/rgb_anon` and `clean_glob: "*/train_ref/*_rgb_ref_anon.png"`. Use the analogous `val_ref` pattern for restoration evaluation.
