mkdir assets
# python3 main.py --n-epochs 10 --do-train
# python main.py --n-epochs 10 --do-train --task custom --reinit_n_layers 3
python main.py --n-epochs 10 --do-train --task supcon --simclr --batch-size 64
# python main.py --n-epochs 10 --do-train --task tune --batch-size 64 --technique 2
# python3 main.py --n-epochs 10 --do-train --task tune --technique 1
