
python -m graphsage.unsupervised_train --prefix example_data/toy-ppi --epochs 20 --max_degree 25 --model graphsage_mean --cuda True --max_total_steps 100


python -m graphsage.supervised_train --prefix example_data/toy-ppi --multiclass True --epochs 1 --max_degree 25 --model graphsage_mean --cuda True --max_total_steps 80
