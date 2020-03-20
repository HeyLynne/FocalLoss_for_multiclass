# focal_for_multiclass

## Introduction
Focal loss is proposed in the paper[Focal Loss for Dense Object Detection](http://arxiv.org/abs/1708.02002). This paper was facing a task for binary classification, however there are other tasks need multiple class classification.
There were few implementation about this task, so I implemented it with a NER task using Albert.


## Prerequisite
- python 3.6
- torch 1.4
## Usage
```
python run_focal_loss.py
```
## Experiments
We use focal loss in Albert model for Ner task, we will public this part of code later, and the result is:

  | Albert  | Albert+Focal loss
---- | ----- | ------  
acc  | 90.792207 | 91.001332 
recall  | 90.145559 | 90.406321 
