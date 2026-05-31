import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import fashion_mnist

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="CNN Fashion MNIST",
    layout="wide"
)

# ---------------------------------
# Load Dataset
# ---------------------------------

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

class_names = [
    'T-shirt',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle Boot'
]

# ---------------------------------
# Sidebar
# ---------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset Overview",
        "Model Evaluation",
        "Prediction"
    ]
)

# ---------------------------------
# HOME
# ---------------------------------

if page == "Home":

    st.title("CNN Fashion MNIST Classifier")

    st.write("""
    This project uses a Convolutional Neural Network (CNN)
    to classify Fashion MNIST images into 10 categories.
    """)

    st.subheader("Sample Images")

    fig, axes = plt.subplots(5, 5, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[i], cmap='gray')
        ax.set_title(class_names[y_train[i]], fontsize=8)
        ax.axis('off')

    st.pyplot(fig)

# ---------------------------------
# DATASET OVERVIEW
# ---------------------------------

elif page == "Dataset Overview":

    st.title("Dataset Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Training Samples", X_train.shape[0])

    with col2:
        st.metric("Testing Samples", X_test.shape[0])

    st.metric("Classes", len(class_names))

    unique, counts = np.unique(y_train, return_counts=True)

    st.subheader("Class Distribution")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(class_names, counts)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Pie Chart")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        counts,
        labels=class_names,
        autopct='%1.1f%%'
    )
    st.pyplot(fig)

# ---------------------------------
# EVALUATION
# ---------------------------------

elif page == "Model Evaluation":

    st.title("Model Evaluation")

    model = load_model("fashion_mnist_model.h5")

    history = np.load(
        "history.npy",
        allow_pickle=True
    ).item()

    y_pred = np.load("y_pred.npy")
    y_test_saved = np.load("y_test.npy")

    accuracy = accuracy_score(y_test_saved, y_pred)
    precision = precision_score(
        y_test_saved,
        y_pred,
        average='weighted'
    )

    recall = recall_score(
        y_test_saved,
        y_pred,
        average='weighted'
    )

    f1 = f1_score(
        y_test_saved,
        y_pred,
        average='weighted'
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{accuracy:.4f}")
    c2.metric("Precision", f"{precision:.4f}")
    c3.metric("Recall", f"{recall:.4f}")
    c4.metric("F1 Score", f"{f1:.4f}")

    st.subheader("Accuracy Curve")

    fig, ax = plt.subplots()

    ax.plot(history['accuracy'], label='Train')
    ax.plot(history['val_accuracy'], label='Validation')

    ax.legend()

    st.pyplot(fig)

    st.subheader("Loss Curve")

    fig, ax = plt.subplots()

    ax.plot(history['loss'], label='Train')
    ax.plot(history['val_loss'], label='Validation')

    ax.legend()

    st.pyplot(fig)

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test_saved,
        y_pred
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    st.pyplot(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y_test_saved,
        y_pred,
        output_dict=True
    )

    st.dataframe(report)

# ---------------------------------
# PREDICTION
# ---------------------------------

elif page == "Prediction":

    st.title("Predict Fashion Item")

    model = load_model("fashion_mnist_model.h5")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file)

        st.image(image, width=200)

        image = image.convert("L")
        image = image.resize((28, 28))

        img = np.array(image)

        img = img / 255.0

        img = img.reshape(
            1,
            28,
            28,
            1
        )

        prediction = model.predict(img)

        pred_class = np.argmax(prediction)

        confidence = np.max(prediction)

        st.success(
            f"Prediction: {class_names[pred_class]}"
        )

        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )

        st.subheader(
            "Prediction Probabilities"
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.bar(
            class_names,
            prediction[0]
        )

        plt.xticks(rotation=45)

        st.pyplot(fig)