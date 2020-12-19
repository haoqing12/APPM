APPM:Neoantigen Presentation Prediction Model

Neoantigen Presentation Prediction Model, named APPM, is a neoantigen predictor build by HLA-peptides mass spectrometry data and convolutional neural network (CNN). Compared to the netMHCpan4.0, our framework demonstrates higher values of area under the ROC curve (AUC) in some HLA alleles.

Dependencies

tensorflow-gpu 1.14

    conda install --channel https://conda.anaconda.org/fwaters tensorflow-gpu==1.14

pandas

sklearn

Usage


Our models (20 availbale alleles) have been trained. If just for prediction neoantigens, you can run the prediction.py directly.
python script/prediction.py [HLA allele] [input_file]
e.g. python script/prediction.py A0101 pep9

If you want to retrain, you can run APPM.py.
python script/APPM.py 0 [HLA allele]
e.g. python script/APPM.py 0 A0101

If you want to continue training, you can run APPM.py.
python script/APPM.py 1 [HLA allele]
e.g. python script/APPM.py 1 A0101
