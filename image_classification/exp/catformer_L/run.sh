work_path=$(dirname $0)
PYTHONPATH=$PYTHONPATH:../../ \
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
    --data-path /home/u000011200011/dataset/ImageNet/ILSVRC2012/ \
    --model CATFormer_small \
    --batch-size 256 \
    --drop-path 0.1 \
    --epoch 300 \
    --dist-eval \
    --output_dir ${work_path}/ckpt \
    2>&1 | tee -a ${work_path}/log.txt
