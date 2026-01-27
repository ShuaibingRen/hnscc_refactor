#!/bin/bash

# 获取脚本所在目录的上一级目录并加入 PYTHONPATH
SCRIPT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
export PYTHONPATH=$PYTHONPATH:${SCRIPT_DIR}/..

# 使用 python -m 模式运行以正确解析包内相对导入
python -m hnscc_refactor.cli "$@"
