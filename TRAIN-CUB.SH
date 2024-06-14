CUDA_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python main.py --dataset CUB \
--cuda --batch_size 32 --nepoch 30 --attri_num 312 \
--pretrain_epoch 10 --pretrain_lr 2e-4 --classifier_lr 1e-5 --manualSeed 3700 \
--xe 1 --attri 1e-4 --regular 0.0005 \
--l_xe 1 --l_attri 1e-2 --l_regular 0.5e-6 --cpt 2e-9 \
--avg_pool \
--all --gzsl \
