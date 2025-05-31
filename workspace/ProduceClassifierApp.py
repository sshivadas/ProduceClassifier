from MyImports import*
from PIL import UnidentifiedImageError
from ProduceClassifierAppUtils import process_streamlit_input_image, load_model

# Streamlit UI
st.set_page_config(page_title="Produce Classifier", layout="centered")
st.title("🥔🍅🧅 Produce Classifier")
st.write("Upload an image of a **potato**, **tomato**, or **onion** to classify it.")
uploaded_file = st.file_uploader("Choose an image file", type=["jpeg", "jpg", "png"])
model = load_model()
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess image
        preprocessed_image =  process_streamlit_input_image(image)

        # Predict
        prediction = model.predict(preprocessed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)

        # Display result
        st.markdown("### 🧠 Prediction")
        st.success(f"**Class:** {predicted_class}")
        st.info(f"**Confidence:** {confidence:.2%}")

    except UnidentifiedImageError:
        st.error("❌ The uploaded file is not a valid image. Please upload a `.jpg`, `.jpeg`, or `.png` file.")

    except Exception as e:
        st.error(f"❌ Error processing the image: {e}")