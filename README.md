# Constrained Clustering

## Overview
This repository contains an implementation of constrained clustering techniques. Constrained clustering incorporates additional constraints (such as must-link and cannot-link constraints) to guide the clustering process, leading to more meaningful and application-specific clustering results.

## Features
- Implementation of constrained clustering algorithms
- Support for must-link and cannot-link constraints
- Evaluation metrics for clustering performance
- Example datasets for experimentation

## Installation
To use this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/mayankyadav0717/constrainedclustering.git
cd constrainedclustering
pip install -r requirements.txt
```

## Usage
To run the constrained clustering algorithm, use the following command:

```bash
python main.py
```

Modify the `config.py` file to adjust clustering parameters and constraints.

## Dependencies
The project requires the following Python libraries:
- NumPy
- Scikit-learn
- Matplotlib
- Pandas

## Example
An example of how to use the clustering algorithm:

```python
from clustering import ConstrainedClustering

# Define your dataset
X = [[1, 2], [2, 3], [5, 8], [8, 8]]

# Define constraints
must_link = [(0, 1)]
cannot_link = [(2, 3)]

# Initialize and fit model
model = ConstrainedClustering(n_clusters=2, must_link=must_link, cannot_link=cannot_link)
model.fit(X)

# Get cluster labels
labels = model.labels_
print(labels)
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with any improvements.

## License
This project is licensed under the MIT License.

## Contact
For any questions, please reach out via GitHub issues.

