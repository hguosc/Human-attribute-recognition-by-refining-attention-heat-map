# HumanAttributeRecognitionByRefiningAttentionHeatMap

## requirements

Install [caffe\_attribute](https://github.com/hguosc/caffe_attribute).

Same steps as installing the original caffe.

## pre-processing

Crop the test images based on the annotated bouding boxes.

## pre-trained models

[Models](https://cse.sc.edu/~hguo/sources/attribute_models.zip)

## run

Use the corresponding model and weights.

```
computeAveragePrecision.m
showVGGHeat.m
```

## Citation

Please cite the following article if it helps your research.

```
  @article{guo2017human,
    title={Human attribute recognition by refining attention heat map},
    author={Guo, Hao and Fan, Xiaochuan and Wang, Song},
    journal={Pattern Recognition Letters},
    volume={94},
    pages={38--45},
    year={2017},
    publisher={North-Holland}
  }
```
