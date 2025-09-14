collection=tvr
visual_feature=i3d_resnet
root_path=/home/jd/桌面/MGAKD/data
#model_dir=tvr-double_kl_exponential_decay_k0.95_loss_scale_weight0.1-2025_09_04_00_03_38
#model_dir=tvr-double_kl_exponential_decay_k0.95_loss_scale_weight0.1-2025_09_07_03_15_53
model_dir=tvr-double_kl_exponential_decay_k0.95_loss_scale_weight0.1-2025_09_13_13_52_47

# # training
python method/eval.py  --collection $collection --visual_feature $visual_feature \
                    --root_path $root_path  --dset_name $collection --model_dir $model_dir
