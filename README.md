# fashion-recommender-system

fashion recommender system methodology description
Fashion recommender systems (FRS) use a variety of methodologies to provide personalized recommendations to users. These systems often combine techniques from content-based filtering, collaborative filtering, and context-aware approaches. Content-based filtering involves recommending items similar to those a user has liked in the past, while collaborative filtering suggests items based on the preferences of similar users. Context-aware approaches incorporate additional information such as user location, time of day, and weather conditions to refine recommendations.

Recent advancements in FRS have seen the integration of deep learning techniques, particularly convolutional neural networks (CNNs) and recurrent neural networks (RNNs), to process visual and textual data. These models are used to extract features from fashion images and text descriptions, which are then used to generate recommendations. For example, a system might take an input image of a fashion item and use CNNs to classify and extract features from the image, which are then used to find similar items in a database (2).
Screenshot 2025-01-02 164248.png
![Uploading Screenshot 2025-05-26 192551.png…]()
![Picture1](https://github.com/user-attachments/assets/a41e3bd1-a090-48d9-8324-a71baaf300ef)
# Page config
st.set_page_config(page_title="Fashion Recommendation", layout="wide")
st.header("👕👚👖👟 Fashion Recommendation System")
![Picture6](https://github.com/user-attachments/assets/a91a6b8c-317a-40de-b7d9-67a33ca7f9d6)

# Load data
Image_features = pkl.load(open('Images_features.pkl', 'rb'))
filenames = pkl.load(open('filenames.pkl', 'rb'))

# Define feature extractor
def extract_features_from_images(img, model):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result / norm(result)
    return norm_result
![Picture5](https://github.com/user-attachments/assets/3916682b-8e37-4f11-be6a-0813f9b0ade0)

# Load model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = tf.keras.models.Sequential([
    base_model,
    GlobalMaxPool2D()
])

# Fit Nearest Neighbors
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)
![Picture4](https://github.com/user-attachments/assets/5ae56959-640f-4a24-b3f5-f1de73c4f873)
# Upload interface
upload_file = st.file_uploader("Upload a clothing image", type=["jpg", "jpeg", "png"])

if upload_file is not None:
    try:
        # Read uploaded image
        img = Image.open(upload_file).convert('RGB')

        st.subheader("📸 Uploaded Image")
        st.image(img, use_column_width=False, width=300)

        # Extract features
        input_img_features = extract_features_from_images(img, model)

![Screenshot 2025-04-08 141043](https://github.com/user-attachments/assets/e50e9d54-5c44-4b92-a2d6-ee9637133bed)
        # Find recommendations
        distances, indices = neighbors.kneighbors([input_img_features])

        st.subheader("🔍 Recommended Items")
        cols = st.columns(5)
        for i, col in zip(indices[0][1:], cols):  # Skip the query image itself
            col.image(filenames[i], use_column_width=True)






        
