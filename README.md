# Eyal Ben Barouch - 318651494  
# Alon Fridental - 209370774  

## Deep Learning Assignment 2 Report  
### Siamese Neural Networks for One-shot Image Recognition  

---

## Introduction  
In this project, we implemented a Siamese network architecture inspired by the paper *"Siamese Neural Networks for One-shot Image Recognition"* to solve the task of verifying whether two faces are identical or not. The network uses two identical branches to extract features from the image pairs, compares them using the L1 distance, and predicts if they match. Along the way, we learned about designing deep learning architectures, experimenting with different configurations, and understanding the impact of data augmentation, regularization, and hyperparameters. This hands-on experience highlighted what works and what doesn’t when training such models.

---

## Data Analysis - LFW-a  
**Train Dataset Statistics:**  
- Total Examples: 2200  
  - Positive Pairs: 1100  
  - Negative Pairs: 1100  

**Test Dataset Statistics:**  
- Total Examples: 1000  
  - Positive Pairs: 500  
  - Negative Pairs: 500  

**Image Size:** (250, 250)  
**Image Mode:** Grayscale  

---

## Data Loading  
- Mount your drive.  
- Save the zip file in your drive inside the folder named `DeepLearning`.  
- Unzip it and allocate the dataset as follows:  
  - Create a folder `lfw2`.  
  - Create a sub-folder called `lfw2` inside `lfwa` and copy all images there.  
  - Create subfolders `train` and `test`, and allocate files accordingly.  

For example:  
`/content/lfw2/test/AJ_Lamas/AJ_Lamas_0001.jpg`

---

## Data Preprocessing  
To enhance robustness and ensure consistency in input size and format, the following preprocessing steps were applied:  

- **Resizing:** All images were resized to 250x250 pixels.  
- **Data Augmentation:**  
  - **Random Horizontal Flip:** Images were flipped horizontally with a random probability.  
  - **Random Rotation:** Images were rotated within a range of ±10 degrees.  
- **Normalization:** Images were normalized with a mean of 0.5 and a standard deviation of 0.5.

---

## Initialization  
While the paper suggests initializing weights with a normal distribution (mean=0, std=0.01), we achieved better results using:  
- **Kaiming He Initialization:** For convolutional layers (optimized for ReLU activations).  
- **Xavier Glorot Initialization:** For fully connected layers (optimized for symmetric activations like Sigmoid).  
- **Batch Normalization Layers:** Initialized with weights of 1.0 and biases of 0.0.  

This improved convergence, stabilized training, and maintained gradient flow, leading to significantly better performance.

---

## Hyperparameters  
Using GridSearch, we found the following hyperparameters to give the best results:  
- **Validation Split:** 20%  
- **Batch Size:** 64  
- **Learning Rate:** 0.01  
- **Momentum:** 0.9  
- **Weight Decay:** 0.001  
- **Max Epochs:** 50  
- **Early Stopping Patience:** 20  
- **Seed:** 42  

---

## Model Architecture  

### 1. Paper-Based Architecture  
- **Input Size:** 250x250  

| Layer | Input Size    | Filters | Kernel Size | Max Pooling | Activation |
|-------|---------------|---------|-------------|-------------|------------|
| 1     | 1x250x250     | 64      | 10x10       | Yes, Stride 2 | ReLU       |
| 2     | 64x48x48      | 128     | 7x7         | Yes, Stride 2 | ReLU       |
| 3     | 128x21x21     | 128     | 4x4         | Yes, Stride 2 | ReLU       |
| 4     | 128x9x9       | 256     | 7x7         | No            | ReLU       |
| 5     | 256x6x6       | -       | -           | -            | Sigmoid    |

- The final output size was reduced from 4096 to 1024 in some experiments, though this did not improve test performance significantly.

---

### 2. Simplified Architecture  
To address overfitting in the paper-based model, we designed a simpler architecture:  

| Layer | Input Size    | Filters | Kernel Size | Max Pooling | Activation |
|-------|---------------|---------|-------------|-------------|------------|
| 1     | 1x250x250     | 32      | 5x5         | Yes, Stride 2 | ReLU       |
| 2     | 32x62x62      | 64      | 3x3         | Yes, Stride 2 | ReLU       |
| 3     | 64x15x15      | 128     | 3x3         | Yes, Stride 2 | ReLU       |
| 4     | 128x4x4       | -       | -           | -            | ReLU       |

This simplified model used dropout (0.3) and batch normalization to improve generalization.

---

## Experiments Comparison  

- **Max Epochs - 15** vs. **Max Epochs - 50**: Longer training improved performance but increased overfitting.  

---

## Error Analysis  

### Paper Model  
The model made acceptable errors, such as misclassifications where even humans might struggle (e.g., images with significant angle differences).  

### Simplified Model  
While reducing overfitting, this model struggled to capture essential details, leading to more severe misclassifications.  

---

## Conclusion  
The Paper Model performed better overall but suffered from overfitting due to its complexity. The Simplified Model reduced overfitting but lacked the capacity to generalize effectively.  

Key lessons include the importance of balancing model complexity with dataset size and exploring stronger regularization, data augmentation, or advanced architectures like attention mechanisms to improve performance. Future research should investigate intermediate architectures or pretrained models.
