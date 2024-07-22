# Self-Organizing Map (SOM) Implementation

This repository contains an improved Python implementation of a Self-Organizing Map (SOM), a type of artificial neural network used for unsupervised learning and data visualization. The improvements are listed below: 
1. For loops are replaced by vector operations to increase time efficiency
2. The code is refactored to increase modularity, productionazation, readability and scalability.
3. A dockerfile is prepared for environment setup.
4. Unittests are written to test correctness of the functions, and to detect and fix bugs early.
5. Variables with Non-ASCII characters in their names are renamed to ensure compatibility with other environments. 

## Usage

To train a SOM and visualize the result, you can use the following code:
```
import numpy as np
import matplotlib.pyplot as plt
from som import SOM

# Set seed
np.random.seed(99)

# Generate data
input_data = np.random.random((10, 3))

# Create and train SOM
som = SOM(input_data, n_max_iterations=100, width=10, height=10, initial_lr=0.1)
som.train()

# Get trained weights
image_data = som.weights

# Save and display the result
fig = plt.subplots()
plt.imsave('100.png', image_data)
plt.imshow(image_data)
```

The produced output by the refactored code and the original code are shown below (using ```seed=99```)
num iterations=100, width=10, height=10     |  num iterations=1000, width=100, height=100     | 
:-------------------------:|:-------------------------:
<img src="notebook/100.png" width=600> | <img src="notebook/1000.png" width=600> |


## Environment Setup

To setup a virtual environment for this project run:
```
docker build -t my-som .
```

