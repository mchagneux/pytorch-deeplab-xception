CUDA_VISIBLE_DEVICES=0 python train.py --backbone mobilenet --lr 0.01 --workers 4 --epochs 40 --batch-size 2 --gpu-ids 0 --checkname deeplab-mobilenet --eval-interval 1 --dataset taco
