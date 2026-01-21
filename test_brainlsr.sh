echo "------------------ Brats2021 FLAIR Dataset ------------------"
python inference.py --task brats21_flair --scale 4 \
    --chop_size 512 --chop_stride 448 --bs 16  \
    --config_path configs/brainsr_DiTM.yaml \
    --ckpt_path weights/brats21_flair.pth \
    -i testdata/brats21_flair/test/lq \
    -r testdata/brats21_flair/test/gt \
    -o results/brats21_flair/test/

echo "------------------ IXI T1 Dataset ------------------"
python inference.py --task ixi_t1 --scale 4 \
    --chop_size 512 --chop_stride 448 --bs 16  \
    --config_path configs/brainsr_DiTM.yaml \
    --ckpt_path weights/ixi_t1.pth \
    -i testdata/ixi_t1/test/lq \
    -r testdata/ixi_t1/test/gt \
    -o results/ixi_t1/test/res

echo "------------------ IXI T2 Dataset ------------------"
python inference.py --task ixi_t2 --scale 4 \
    --chop_size 512 --chop_stride 448 --bs 16  \
    --config_path configs/brainsr_DiTM.yaml \
    --ckpt_path weights/ixi_t2.pth \
    -i testdata/ixi_t1/test/lq \
    -r testdata/ixi_t1/test/gt \
    -o results/ixi_t1/test/res