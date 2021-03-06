# NeurIPS 2021 Anonymous Repo 

Suppementary Notebooks and Data in support of our paper:

**_Post-mortem on a deep learning contest: a Simpson’s paradox and the complementary roles of size metrics versus shape metrics_**


## Reproducing the WeightWatcher and Baseline Results

 All calculations have been computed using Google Colab, 
 Contest Data, with pre-trained models,  must be downloaded first (see below)

For WeightWatcher results, see:
<pre>
WWContestAnalysis-Task1.ipynb
WWContestAnalysis-Task2.ipynb
</pre>

For other results, see the notebooks in  colab-notebooks/, including

<pre>
NeurIPSContest_Baseline.ipynb  
NeurIPSContest_Augmented.ipynb 
NeurIPSContest_W_Winit.ipynb
NeurIPSContest_Sharpness.ipynb 
NeurIPSContest_SVD20.ipynb    
</pre>
<hr>


## Reruning the Data Analysis Jupyter Notebooks

 All figures and tables in the paper can be reproduced with the Jupyter Notebooks

 (You do not need to download the contest data to reproduce the notebooks or figures in the paper)

#### Descriptive Plots

 -[Fig1.ipynb](Fig1.ipynb):  Figure 1, different ESDs 

 -[WeightWatcher-VGG-TrapPlots.ipynb](WeightWatcher-VGG-TrapPlots.ipynb):  Figure 4, Correlation Traps  (and some of Fig 1)

#### Data Analysis
 
 Use papermill to execute all notebooks, generate all plots for paper

 See Utils.IMG_DIR   to set where images are created

 <pre>
 notebooks/run.sh
</pre>

<hr>


### Contest Data: Pretrained Models 

The original code and data can be downloaded from the [Contest Download Link](https://competitions.codalab.org/competitions/25301#learn_the_details-get_starting_kit)

If this is no-longer available, you can download from this [Google Drive link](https://drive.google.com/drive/folders/1kj3xqZctFOAR0Zw3PfE2jx9rtX86QnmO?usp=sharing)


#### Contest Data Directory Contents
 
 starting_kit/public_data/
 -  input_data:  model configs, initial and pretrained weights
 -  reference_data:  model definitions and training and test accuracies    *included in this repo*

<pre>

public_data/input_data/

 task1_v4/model_24 ... model_735/

 task2_v1/model_239 .. model_1011/

     config.json  
     weights_init.hdf5
     weights.hdf5

public_data/reference_data/

   task1_v4/model_configs.json 
   task2_v1/model_configs.json 

</pre>


<hr>

### Results Data Folder Contents

#### Contest training and test data for baseline and comparisons
<pre>
 ./public_data/reference_data/...  (see above)
</pre>


#### WeightWatcher (WW) Results for each model (details dataframe in CSV format)

WW Metrics include:

  - LogSpectralNorm
  - Alpha
  - WeightedAlpha
  - Alpha KS Distance 
  - LogFrobeniusNorm
  - LogShattenNorm

<pre>
 ./results/
    task1_v4/model_xxx.csv    WeightWatcher Details dataframe, in csv format
    task2_v1/model_xxx.csv
</pre>

#### Other Results for Data Dependent Metrics

   - W_distance:  Distance from initial weight matrics

   - sharpness: Sharpness Transform (provided as contest baseline)
   - ww_svd10:  Spectral (SVD) Smoothing keeping 10% of all (tail) eigenvectors
   - ww_svd20:   Spectral (SVD) Smoothing keeping 20% ...

Predicted test accuracies for all models stored in a single file:
<pre>
 ./results/XXX/
    task1_v4.predict   
    task2_v1.predict  
</pre>
















