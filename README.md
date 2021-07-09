# MSDNet with Learning to Cascade
This is a pytorch implementation of the following paper [[AAAI]](https://ojs.aaai.org/index.php/AAAI/article/view/16900) [[arXiv]](https://arxiv.org/abs/2104.09286):  
```
@inproceedings{enomoto2021ltc,
  title={Learning to Cascade: Confidence Calibration for Improving the Accuracy and Computational Cost of Cascade Inference Systems},
  author={Shohei Enomoto and Takeharu Eda},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  pages={7331--7339},
  year={2021}
}
```
Please read license.txt before reading or using the files.  

## Dependencies:

+ Python3
+ PyTorch >= 0.4.0

## Network Configurations

#### Train an MSDNet (block=2, step=2, base=4) on CIFAR-100: 

```
python3 main.py --data-root /PATH/TO/CIFAR100 --data cifar100 --save /PATH/TO/SAVE \
                --arch msdnet --batch-size 64 --epochs 300 --nBlocks 2 \
                --stepmode lin_grow --step 2 --base 4 --nChannels 16 \
                -j 16 --use-valid --seed 0
```

#### Train an MSDNet (block=2, step=2, base=4) with LtC(w=1.5, C=0.5) on CIFAR-100: 

```
python3 main.py --data-root /PATH/TO/CIFAR100 --data cifar100 --save /PATH/TO/SAVE \
                --arch msdnet --batch-size 64 --epochs 300 --nBlocks 2 \
                --stepmode lin_grow --step 2 --base 4 --nChannels 16 \
                -j 16 --use-valid --seed 0 --ltc --w 1.5 --C 0.5
```

#### Test an MSDNet (block=2, step=2, base=4) on CIFAR-100:

```

python3 main.py --evalmode dynamic --batch-size 128 \
                --evaluate-from /PATH/TO/save_models/model_best.pth.tar \
                --arch msdnet --data cifar100 --data-root /PATH/TO/CIFAR100 --nBlocks 2 --step 2 --base 4 \
                --nChannels 16 --stepmode lin_grow --use-valid --seed 0
```
