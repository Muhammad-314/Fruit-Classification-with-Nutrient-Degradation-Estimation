The project is implemented using a modular, three-part architecture: a data augmentation pipeline, a deep learning training script, and a graphical user interface (GUI) for real-time prediction and user management. The entire system is built primarily on Python, leveraging the power of PyTorch for the core machine learning tasks and Tkinter for the interactive user application.

1. Dataset Generation and Augmentation
Given the absence of a large, pre-labelled, multi-stage fruit dataset, the system employs aggressive data augmentation to artificially expand a smaller collection of source images (including some AI-generated images) into a comprehensive dataset.
Data Structure and Classes
The project classifies four fruits, each with a defined number of ripeness stages:
 Banana: 9 stages
 Orange: 5 stages
 Mango: 4 stages
 Apple: 3 stages
Augmentation Methods
1. Keras ImageDataGenerator: This method is applied aggressively to generate 100 training images and 20 validation images per source image. Key augmentation parameters include:
o Geometric Transformations: Rotation (+ 40 %), Shifts (horizontal/vertical + 30%), Shear (+ 30 %), and Zoom (+ 30%).
o Colour/Lighting Variations: Horizontal/Vertical Flip, Brightness (50% to 150%), and Channel Shift .
2. Manual PIL Augmentations : This provides complementary variations focusing on different methods, generating 30 training images and 5 validation images per source image. It incorporates random rotation, scaling, and brightness/contrast adjustments using the PIL library.
The combined approach ensures a diverse and extensive training set, where each source image contributes 130 training examples and 25 validation examples to its corresponding class.

2. Model Training
It handles the core deep learning task: training a state-of-the-art Convolutional Neural Network (CNN) model on the augmented dataset.
Hardware Acceleration
12
The script features a strict GPU-enforced training environment, prioritising acceleration for efficiency. The get_device function implements a smart device detection hierarchy :
1. NVIDIA CUDA: Utilised if an NVIDIA GPU is present.
2. AMD DirectML: Used for AMD/Intel GPUs if the torch-directml package is installed.
3. No CPU Fallback: If neither GPU is available, the training is halted to ensure high-performance execution.
Model Architecture and Configuration
 Base Model: ResNet-18 is used as the backbone architecture, initialised with pre-trained weights to benefit from transfer learning.
 Output Layer: The final fully connected layer is modified to output 21 classes (matching the total number of fruit stage classes).
 Training Configuration:
o Loss Function: Cross Entropy Loss (Standard for multi-class classification).
o Optimiser: Adam optimizer with a learning rate of 0.001.
o Scheduler: Learning rate Scheduler to decrease the learning rate by a factor of 0.1 every 7 epochs, promoting convergence.
o Epochs: Trained for a total of 50 epochs.
Data Loading
The augmented data is loaded via PyTorch ImageFolder and DataLoader, applying specific transforms for training (including rotation, flip, and colour jitter) and validation (resizing and normalisation). The process handles datasets that are pre-split into train/val directories or performs an 80/20 split automatically if raw data is provided.
The best model weights, based on validation accuracy, are saved to a file named fruit_resnet_model.pth for deployment.

3. Prediction and Deployment GUI
The prediction.py script runs the deployed model within a Tkinter-based GUI, providing a user-friendly, feature-rich application for live fruit analysis. The application is divided into three main functional systems .
13
3.1 Fruit Prediction Engine (FruitPredictor)
This is the core of the application, responsible for running the AI model and nutrient analysis .
 Model Loading: The saved PyTorch model is loaded, ensuring it runs on the best available device (CUDA, DirectML, or CPU).
 Prediction Logic:
1. PyTorch Inference: The preprocessed image is fed to the ResNet-18 model to get the Top classification predictions.
2. Computer Vision (CV) Rot Check: A supplementary OpenCV check (is_rotten method) uses colour space analysis (HSV and LAB) to detect excessive brown, dark, or mould areas in the image, providing an independent measure of decay.
 Nutrient Database: literature derived NUTRIENT TABLES provides stage-specific nutritional information (sugar, Vitamin C, fibre, calories) and benefits for displaying alongside the prediction.
 The confidence calculation method uses a standard PyTorch implementation involving the Softmax function:
The confidence score represents the model's certainty that the input image belongs to the predicted ripeness stage (e.g., banana_stage_5).
1. Model Output (Logits)
The PyTorch model processes the input image and outputs a set of raw, unnormalized scores called logits for each possible class (ripess stage).
2. Test-Time Augme
 Input Duplication: The image is processed twice: once as the Original image and once as a Horizontally Flipped image.
 Batch Prediction: Both images are stacked into a batch and passed through the model.
 Probability Calculation (Softmax): The raw logits from both the Original and Flipped predictions are converted into probabilities using the Softmax function. This ensures the scores for each prediction sum to 1.
3. TTA Averaging
The core of the TTA technique is to average the probabilities from the augmented (flipped) and original inputs:
4. Ranking and Final Confidence
The avg_probs vector is then sorted, and the highest value is selected as the final prediction's confidence score:
 The class with the highest average probability is determined as the final predicted label_1 (e.g., banana_stage_5).
 The actual value of this highest probability is assigned as the final conf_1.
3.2 User Interface
The main application screen features :
 Input: Users can load images via file browsing or live camera feed (using OpenCV for capture).
 Prediction Display: Results are presented in a colour-coded card format, showing the predicted stage name, confidence, and a full nutritional breakdown when the fruit is not rotten.
 Fruit Selection: A dropdown allows the user to filter predictions for a specific fruit (e.g., only classify for 'banana' stages) or use the default 'auto' mode.
