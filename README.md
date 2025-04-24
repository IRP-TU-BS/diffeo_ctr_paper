# Diffeomorphic-Mapping for CTR Force Sensing
This repository contains the implementation of a novel data-driven force sensing approach for Concentric Tube Continuum Robots (CTRs) using diffeomorphic-mapping-based transfer learning.

### Overview

Concentric tube continuum robots (CTRs) are ultra-flexible active catheters used in minimally invasive medical procedures. Force sensing is critical for patient safety and surgical precision, but integrating hardware sensors within these slender robots is challenging due to space constraints.
Our approach addresses this challenge by:

- Using a data-driven method that estimates tip contact forces based on tube deflection
- Employing diffeomorphic mappings to transfer knowledge between different tube configurations
- Drastically reducing the need for extensive training data for each new robot configuration

This method allows us to train a neural network on straight tube configurations and then apply this knowledge to pre-curved tubes, eliminating the need for retraining with each new tube geometry.
Key Features

- Transfer learning approach for force estimation in CTRs
- Handles multiple tubes with varying curvatures
- Supports tube rotation and elongation
- Achieves RMSE of 0.0287 ± 0.00461 N in testing
- Effective for curvatures κ ≤ 7 1/m with high estimation accuracy


### Installation

To install the required dependencies, use the following command:

```bash
pip install -r requirements.txt
```

### Code Location

All the code for this project is located in the `notebooks` folder. The code is designed to be used with Jupyter Notebook.

### Using Jupyter Notebook

To use Jupyter Notebook, follow these steps:

1. Launch Jupyter Notebook by running:
    ```bash
    jupyter notebook
    ```
2. Navigate to the `notebooks` folder in the Jupyter interface.
3. Open the desired notebook file to explore and run the code.

For more information on Jupyter Notebook, visit the [official documentation](https://jupyter.org/documentation).