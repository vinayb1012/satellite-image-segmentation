# Satellite Image Segmentation

**Goal**: To study the differences between using multispectral and RGB channels for satellite image segmentation.

![Samples](images/sample1.png)
![Samples](images/sample2.png)
![Samples](images/sample3.png)

## Install requirements

```python
pip install requirements.txt
```

## Training

```python
python train.py --session-name rgb --channels rgb --epochs 100
```

## Evaluate

```python
python evaluate.py --channels rgb --model output/rgb/models/best.pth
```

