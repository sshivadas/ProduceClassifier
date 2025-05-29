# ProduceClassifier
This project builds a computer vision model to classify three common vegetables — onion, potato, and tomato — and distinguish them from background noise such as Indian market scenes. The aim is to assist in automatically recognizing vegetable items from photos, which can be useful for applications like automated checkout systems or inventory management.
Dataset:-
This dataset contains images of the following food items: 
1. Vegetables- onion, potato and tomato
2. Noise-Indian market
The images in this dataset were scraped from Google.
https://drive.google.com/file/d/1clZX-lV_MLxKHSyeyTheX5OCQtNCUcqT/view?usp=sharing

This dataset contains a folder train, which has a total of 3135 images, split into four folders as follows:
•	Tomato : 789
•	Potato : 898
•	Onion : 849
•	Indian market : 599
This dataset contains another folder test which has a total of 351 images, split into four folders
•	Tomato : 106
•	potato : 83
•	onion : 81
•	Indian market : 81
Raw image data was augmented to improve generalization.

Model Architectures:
Started with a custom Convolutional Neural Network (CNN).
Explored transfer learning with VGGNet, ResNet, and MobileNet.
ResNet-50 achieved the best performance across key metrics.

Training Strategy:
Implemented using TensorFlow/Keras.
Used Adam optimizer, batch normalization, dropout, early stopping, and GlobalAveragePooling.
TensorBoard callbacks were used for logging and visualization.

Evaluation Metrics:
Accuracy, Precision, Recall, and F1 Score.
Plotted training vs. validation accuracy/loss for performance tracking.

Challenges & Considerations:
Dataset is imbalanced (no resampling/weighting applied yet).

Model Deployment:
Folder Structure:
produce_classifier/
├── flask_api/
│   ├── app.py                # Flask backend
│   └── model.h5              # Trained ResNet50 model
├── streamlit_app/
│   └── app.py                # Streamlit frontend
├── requirements.txt          # Combined requirements
├── README.md                 # Project overview
└── runtime.txt (optional for Render)
Deployment on Render (Free Tier)