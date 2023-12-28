# SpikingViT
The implement of spikingVit for object detection of event-based datasets.
![1703716888213](figs/293152151-c68aa765-f58f-4434-8121-564dfd702b18.png)
![1703718305802](figs/293155252-0d45bd1f-7788-4054-ac08-474e4641fcf7.png)
![1703718363225](figs/293155375-68793247-f2b3-4145-86a4-b08fc4671fcc.png)


## Conda Installation
```bash
conda create -n spikingVit python=3.10
conda activate spikingVit

pip install -r requirements.txt
```

## Runing scripts
We recommend you runing our code on **Linux** platform, as there are some error about multiprocessing package.
```bash
sudo chmod ./train_gen1/4.sh
./train_gen1/4.sh
```

## Validataion scripts
```bash
sudo chmod ./val_gen1/4.sh
./val_gen1/4.sh
```

## Citation
If you find our work helpful for your research, please consider citing the following BibTeX entry.
```Tex

```
And don't forget to cite our acknowledgment project.

## Code Acknowledgments
This project has used code from the following projects:
- [RVT](https://github.com/uzh-rpg/RVT) for their most of code and dataset preprocessing.
