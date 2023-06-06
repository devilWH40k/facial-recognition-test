# Face verification test

Project created in research purposes to evaluate a couple of face recognition models on given dataset.

## Setup
---

Set up virtual env:
```
python -m venv venv
source venv/bin/activate
```
if you have windows use:
```
.\venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

## Run experiments
---
Make sure to have a dataset in the `./dataset` location,
which contains subdirectories representing speakers, which
contain image files with following extensions: "jpg", "png", "jpeg", "jfif".

The the experiments can be run using the following command:

```
python main.py
```

The output of the program will be a table in the console,
containing all experiments scores and information.

**NOTE**: script also creates a "dataset.obj" file for caching a result of generated dataset pairs, so if you want to update the pairs simply delete this file.