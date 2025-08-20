# Samsung EnnovateX 2025 AI Challenge Submission

- **Problem Statement** - Classify User Application Traffic at the Network in a Multi-UE Connected Scenario
Applications are affected differently under varying traffic conditions, channel states, and coverage scenarios. If the traffic of each UE can be categorized into broader categories, such as Video Streaming, Audio Calls, Video Calls, Gaming, Video Uploads, browsing, texting etc. that can enable the Network to serve a differentiated and curated QoS for each type of traffic. Develop an AI model to analyze a traffic pattern and predict the application category with high accuracy.
- **Team name** - *(As provided during the time of registration)*
- **Team members (Names)** - *Member 1 Navya*


### Project Artefacts

# Technical Details

## Source Code Organization

### Source (`src/`)
- `model_training2.py`: Main pipeline implementation
- Location: `src/model_training2.py`
- Dependencies: Listed in `requirements.txt`

### Documentation (`docs/`)
- Technical documentation is available in the `docs/` directory:
  - `overview.md`: Project architecture and approach
  - `installation.md`: Setup instructions
  - `pipeline_details.md`: Implementation details
  - `usage.md`: Usage guide and examples
  - `troubleshooting.md`: Common issues and solutions

## Models

### Models Used
- RandomForestClassifier from scikit-learn
- HistGradientBoostingClassifier from scikit-learn
- VotingClassifier for ensemble creation

No pre-trained models from Hugging Face were used in this project.

## Datasets

### Dataset Used
- Name: Combined Traffic Classification Dataset
- Size: 148,088 samples
- Features: 24
- Classes: 14
- Location: `data/combined_data.zip`
- Format: CSV

Note: The dataset contains proprietary information and is not publicly shared. Users need to provide their own dataset in the same format for using this pipeline.

### Dataset Format Requirements
```python
Required columns:
- 'class1': Target variable (14 classes)
- 23 feature columns (numerical or categorical)
```

### Data Privacy
The original dataset contains sensitive information and is not published. Users should ensure they have appropriate permissions for their data usage.

## Project Structure
```
ML-MultiClass-MaxAccuracy-Pipeline/
├── src/
│   ├── model_training2.py
│   └── README.md
├── docs/
│   ├── feature.md
│   ├── approach.md
│   ├── implementation.md
│   ├── stack.md
│   ├── user.md
│   └── troubleshooting.md
├── data/
│   ├── README.md
│   └── combined_data.zip
├── requirements.txt
├── README.md
└── LICENSE
```

### Attribution 

In case this project is built on top of an existing open source project, please provide the original project link here. Also, mention what new features were developed. Failing to attribute the source projects may lead to disqualification during the time of evaluation.


# Technical Details

## Source Code Organization

### Source (`src/`)
- `model_training2.py`: Main pipeline implementation
- Location: `src/model_training2.py`
- Dependencies: Listed in `requirements.txt`

### Documentation (`docs/`)
- Technical documentation is available in the `docs/` directory:
  - `approach.md`: Project architecture and methodology
  - `feature.md`: Key features and capabilities
  - `implementation.md`: Pipeline implementation details
  - `stack.md`: Technical stack and dependencies
  - `user.md`: Usage guide and examples
  - `troubleshooting.md`: Common issues and solutions

## Models

### Models Used
- RandomForestClassifier from scikit-learn
- HistGradientBoostingClassifier from scikit-learn
- VotingClassifier for ensemble creation

No pre-trained models from Hugging Face were used in this project.

## Datasets

### Dataset Used
- Name: Combined Traffic Classification Dataset
- Size: 148,088 samples
- Features: 24
- Classes: 14
- Location: `data/combined_data.zip`
- Format: CSV

Note: The dataset contains proprietary information and is not publicly shared. Users need to provide their own dataset in the same format for using this pipeline.

### Dataset Format Requirements
```python
Required columns:
- 'class1': Target variable (14 classes)
- 23 feature columns (numerical or categorical)
```

### Data Privacy
The original dataset contains sensitive information and is not published. Users should ensure they have appropriate permissions for their data usage.

## Project Structure
```
ML-MultiClass-MaxAccuracy-Pipeline/
├── src/
│   ├── model_training2.py
│   └── README.md
├── docs/
│   ├── feature.md
│   ├── approach.md
│   ├── implementation.md
│   ├── stack.md
│   ├── user.md
│   └── troubleshooting.md
├── data/
│   ├── README.md
│   └── combined_data.zip
├── requirements.txt
├── README.md
└── LICENSE
```
