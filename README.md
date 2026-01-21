<div align="center">
<h2>Combining Transformer and Mamba Diffusion Structures for Super-Resolution of Brain MRI</h2>
<div>
    <a href='https://github.com/kunncheng'>Qiong Zhang* <sup>1</sup></a>&emsp;
</div>
</div>


## 🌈 Training
### Datasets
The training data comprises [BraTs2021-FLAIR](https://pan.baidu.com/s/13d_KSRxGe3qUtEsg7jupVQ?pwd=j56k), [IXI](https://pan.baidu.com/s/1LFLATBJ6ybRXuNaawlU5Uw?pwd=1ih2)


### Training Scripts
**Brain MRI Super-resolution**
```
torchrun --standalone --nproc_per_node=8 --nnodes=1 main.py --cfg_path configs/brainsr_DiTM.yaml --save_dir ${save_dir}
```



## 🚀 Inference and Evaluation
**Brain MRI Super-resolution**
```
bash test_realsr.sh
```

## ❤️ Acknowledgement
We sincerely appreciate the code release of the following projects: [DiT-SR](https://github.com/kunncheng/DiT-SR),[DiM](https://github.com/tyshiwo1/DiM-DiffusionMamba),[ResShift](https://github.com/zsyOAOA/ResShift), [DiT](https://github.com/facebookresearch/DiT), [FFTFormer](https://github.com/kkkls/FFTformer), [SwinIR](https://github.com/JingyunLiang/SwinIR), [SinSR](https://github.com/wyf0912/SinSR), and [BasicSR](https://github.com/XPixelGroup/BasicSR).
