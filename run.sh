mkdir assets

# Darren
python main.py --n-epochs 10 --do-train --task tune --technique 1 --reinit_n_layers 2

# Brandon
python main.py --n-epochs 10 --do-train --task tune --technique 3
python main.py --n-epochs 10 --do-train --task supcon --simclr

# Nathan
python main.py --n-epochs 10 --do-train --task supcon