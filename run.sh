mkdir assets

# Darren
python main.py --n-epochs 10 --do-train --task tune --technique 2 --reinit_n_layers 3

# Brandon
python main.py --n-epochs 10 --do-train --task tune --technique 3

# Nathan
python main.py --n-epochs 10 --do-train --task supcon