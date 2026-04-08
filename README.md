## 环境搭建
运行脚本：./env.sh
## 数据集构建
tgb/datasets/thgl_{数据集名}_subset下，依次运行get\_{数据集}_subset.ipynb以及thgl\_{数据集}_subset_ns_gen.py

## 使用示例
```
python src/main.py --exper_name example --dataset thgl-forum-subset --use_onehot_node_feats --use_graph_structure --model hetero_sthn --use_riemannian_structure
```
