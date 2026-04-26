#!/bin/bash

# 通过输入参数选择
if [ $# -eq 0 ]; then
    echo "用法: $0 [train|close_eval|showcfg]"
    exit 1
fi

case "$1" in
    train)
        python -m il.train
        ;;
    close_eval)
        python -m il.eval
        ;;
    showcfg)
        python -m il.train --cfg job
        ;;
    *)
        echo "未知选项: $1"
        echo "可用选项: default, batch32, showcfg"
        exit 1
        ;;
esac
