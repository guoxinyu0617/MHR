#weibo
python train.py --dataset weibo --dim 64 --lr 0.0006 --num-layers 3 --cuda 0 --margin 8.0 --c None --split-seed 13
#twitter
python train.py --dataset twitter --dim 16 --lr 0.0003 --num-layers 2 --cuda 0 --margin 5.0 --c None --split-seed 246813
# pheme
python train.py --dataset pheme --dim 32 --lr 0.0003 --num-layers 3 --dropout 0.2 --cuda 0 --margin 1.0 --c None --split-seed 12345
