collection=activitynet
visual_feature=i3d
root_path=/home/jd/桌面/MGAKD/data
#model_dir=activitynet-double_kl_exponential_decay_k0.95_loss_scale_weight0.1-2025_07_09_21_21_08
model_dir=activitynet-double_kl_exponential_decay_k0.95_loss_scale_weight0.1-2025_09_02_10_11_40
#model_dir=activitynet-double_kl_exponential_decay_k0.95_loss_scale_weight0.1-2025_09_03_21_07_32
#model_dir=activitynet-double_kl_exponential_decay_k0.95_loss_scale_weight0.1-2025_09_04_03_25_02
#model_dir=activitynet-double_kl_exponential_decay_k0.95_loss_scale_weight0.1-2025_09_04_11_25_30
#activitynet-double_kl_exponential_decay_k0.95_loss_scale_weight0.1-2024_06_15_19_50_30
#activitynet-double_kl_exponential_decay_k0.95_loss_scale_weight0.1-2024_05_27_10_49_18


#tvr-double_kl_exponential_decay_k0.95_loss_scale_weight0.1-2023_12_15_15_42_36
# # training
python method/eval.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --model_dir $model_dir


