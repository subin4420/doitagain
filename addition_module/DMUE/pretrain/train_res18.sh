CUDA_VISIBLE_DEVICES=0,1 python train_mv_softmax.py \
    --backbone 'resnet18' \
    --data_root '/path/to/your/cropped_msra/' \
    --train_file '/path/to/your/msra_train_file.txt' \
    --out_dir './checkpoints/out_dir_res18' \
    --lr 0.1 \
    --step '10, 13, 16' \
    --epochs 18 \
    --print_freq 200 \
    --batch_size 512 \
    --feat_dim 512
