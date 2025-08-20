# Installation Guide

## Prerequisites
- Python 3.x
- 16GB+ RAM recommended
- Multi-core processor (code uses parallel processing)

## Setup Steps

1. Clone the repository:
```bash
git clone [https://github.com/yourusername/ML-MultiClass-MaxAccuracy-Pipeline.git](https://github.com/Navya12341/Traffic-Predictor
cd ML-Traffic Predictor
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare data directory:
```bash
cd data
unzip combined_data.zip
```

## Mac-Specific Notes
To prevent sleep during long training runs:
```bash
caffeinate -i python src/model_training2.py > training_log.txt 2>&1
```
