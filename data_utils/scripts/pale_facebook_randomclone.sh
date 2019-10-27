python -m data_utils.pale_random_clone --input $HOME/dataspace/graph/pale_facebook/graphsage \
    --output1 $HOME/dataspace/graph/pale_facebook/random_clone/ \
    --output2 $HOME/dataspace/graph/pale_facebook/random_clone/ \
    --prefix pale_facebook \
    --alpha_s 0.9 --alpha_c 0.9


python -m data_utils.pale_random_clone --input $HOME/dataspace/graph/ppi/sub_graph/graphsage \
    --output1 $HOME/dataspace/graph/ppi/random_clone/ \
    --output2 $HOME/dataspace/graph/ppi/random_clone/ \
    --prefix ppi \
    --alpha_s 0.5 --alpha_c 0.9


python -m data_utils.pale_random_clone --input $HOME/dataspace/graph/ppi/sub_graph/graphsage \
    --output1 $HOME/dataspace/graph/ppi/random_clone/ \
    --output2 $HOME/dataspace/graph/ppi/random_clone/ \
    --prefix ppi \
    --alpha_s 0.6 --alpha_c 0.5




python -m data_utils.pale_random_clone --input $HOME/dataspace/graph/ppi/sub_graph/graphsage \
    --output1 $HOME/dataspace/graph/ppi/random_clone/ \
    --output2 $HOME/dataspace/graph/ppi/random_clone/ \
    --prefix ppi \
    --alpha_s 0.6 --alpha_c 0.9


python -m data_utils.pale_random_clone --input $HOME/dataspace/graph/ppi/sub_graph/graphsage \
    --output1 $HOME/dataspace/graph/ppi/random_clone/ \
    --output2 $HOME/dataspace/graph/ppi/random_clone/ \
    --prefix ppi \
    --alpha_s 0.9 --alpha_c 0.9