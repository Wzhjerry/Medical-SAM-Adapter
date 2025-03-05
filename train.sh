# python train.py -net sam -mod sam_adpt -exp_name vessel_new -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -val_freq 1 -dataset vessel -data_path custom
# python train.py -net sam -mod sam_adpt -exp_name od_2 -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -val_freq 1 -dataset od -data_path custom
python train.py -net sam -mod sam_adpt -exp_name oc_new -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -val_freq 1 -dataset oc -data_path custom
python train.py -net sam -mod sam_adpt -exp_name ex_new -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -val_freq 1 -dataset ex -data_path custom
python train.py -net sam -mod sam_adpt -exp_name he_new -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -val_freq 1 -dataset he -data_path custom

python val.py -net sam -mod sam_adpt -exp_name vessel_new_eval -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -val_freq 1 -dataset relabel -data_path custom -weights ./logs/vessel_new/Model/best_dice_checkpoint.pth
python val.py -net sam -mod sam_adpt -exp_name od_2 -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -val_freq 1 -dataset relabel -data_path custom -weights ./logs/od_2/Model/best_dice_checkpoint.pth
python val.py -net sam -mod sam_adpt -exp_name oc_new_eval -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -val_freq 1 -dataset relabel -data_path custom -weights ./logs/oc_new/Model/best_dice_checkpoint.pth
python val.py -net sam -mod sam_adpt -exp_name ex_new_eval -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -val_freq 1 -dataset relabel -data_path custom -weights ./logs/ex_new/Model/best_dice_checkpoint.pth
python val.py -net sam -mod sam_adpt -exp_name he_new_eval -sam_ckpt ./checkpoint/sam/sam_vit_b_01ec64.pth -image_size 1024 -b 2 -val_freq 1 -dataset relabel -data_path custom -weights ./logs/he_new/Model/best_dice_checkpoint.pth