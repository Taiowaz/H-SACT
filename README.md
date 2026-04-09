## Environment Setup
chmod +x env.sh
Run script: ./env.sh

## Dataset Construction
In tgb/datasets/thgl_{dataset_name}_subset, run the following in order:
1. get_{dataset}_subset.ipynb
2. thgl_{dataset}_subset_ns_gen.py

## Usage Example
```
python src/main.py --exper_name example --dataset thgl-forum-subset --use_onehot_node_feats --use_graph_structure --model hetero_sthn --use_riemannian_structure
```