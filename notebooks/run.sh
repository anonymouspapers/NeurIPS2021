#!/usr/bin/env bash

papermill WWandBaselineResults.ipynb  WWandBaselineResults.out.ipynb 

for m in W_distance D sharpness svd10 svd20
do
    papermill AnalyzeResults.ipynb  AnalyzeResults.$m.ipynb -p METRIC $m
done

papermill NormBasedMetrics.ipynb NormBasedMetrics.out.ipynb

papermill DataDependentMetrics.ipynb DataDependentMetrics.out.ipynb

papermill MakeTable5.ipynb MakeTable5.out.ipynb


