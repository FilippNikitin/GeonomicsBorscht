# Relating epigenetics to 3D genomic structure for single-cell sequencing modalities

>**Introduction**
Different data modalities provide different perspectives on a population of cells, and their
integration is critical for studying cellular heterogeneity and its function. 
A combination of different analytical tasks (e.g., multi-modal integration and cross-modal
analysis) is required to comprehensively understand such data, inferring how gene regulation
drives biological diversity and functions. This study aims to investigate the chromatin structure
landscape of genomic regions by comparing cell-type cluster-aggregated scHiC profiles and inferring
their correlation with epigenetic methylation. To explore the relationship between chromatin conformation and methylation data
modalities (scHiC and sn-methylome) we've designed a graph NN-based method that predicts the
methylation level (range between 0/1) for each genomic bin using HiC information. We use the
mouse brain dataset, which has co-assayed (sn-m3c-seq) signals for HiC and CpG/CpH methylation
levels for single cells. 

### Model Architecture

<div>
<img src=data/model_pipeline.png width="95%">
</div>

We propose a neural network to predict methilation level from HiC data. The main stages of this neural network 
are node initialization, graph convolution neural network, and fully-connected neural network. 

### Run Code

To train the model:
```bash
python train.py experiment/config.yaml
```

To test the model:

```bash
python train.py experiment/config.yaml --test --ckpt_path path_to_checkpoint
```

### Installation
```bash
conda create -n borscht python=3.10
conda activate borscht
pip install -r requirements.txt
```