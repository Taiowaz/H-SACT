# filepath: [env.sh](http://_vscodecontentref_/0)
#!/bin/bash
set -e  # 任何命令失败都停止脚本

conda create -n hsact python=3.10 -y
conda run -n hsact pip install -e .

# 非5090
conda run -n hsact pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
conda run -n hsact pip install torch-sparse torch_geometric torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 5090 (需要时取消注释)
# conda run -n hsact pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
# conda run -n hsact pip install torch-sparse torch_geometric torch_scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

conda run -n hsact pip uninstall numpy -y
conda run -n hsact conda install numpy==1.23.5 pandas scikit-learn -y
conda run -n hsact pip install pytz clint torchmetrics geoopt

echo "Installation completed successfully!"
