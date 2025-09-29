from flask import Flask, render_template, request, jsonify
import pickle
import re
import numpy as np
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# -------------------------
# Setup NLTK inside venv
# -------------------------
nltk_data_path = os.path.join(os.path.dirname(__file__), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

# Add custom path for NLTK data (useful in packaged environments)
nltk.data.path.append(nltk_data_path)
# Ensure necessary NLTK components are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('stopwords', download_dir=nltk_data_path)

# -------------------------
# Flask App
# -------------------------
app = Flask(__name__)

# Load models
# Ensure the file paths are correct relative to app.py
try:
    model = pickle.load(open('classifier.pkl', 'rb'))
    w2v_model = Word2Vec.load('word2vec.model')
    print("Models loaded successfully.")
except FileNotFoundError:
    print("ERROR: One or more model files (classifier.pkl or word2vec.model) not found.")
    model = None
    w2v_model = None

# -------------------------
# Preprocessing
# -------------------------
def preprocess_text(text):
    """Cleans and tokenizes text for prediction."""
    if not text:
        return []
    # Remove all non-alphabetic characters and replace with space
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    
    # Use word_tokenize for better handling, fallback to split if tokenization fails
    try:
        tokens = word_tokenize(review)
    except:
        tokens = review.split()
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

# -------------------------
# Keyword fallback (Further Enhanced for FAKE detection)
# -------------------------
def keyword_fallback(text):
    """
    Rule-based classification used when the ML model has no known words.
    
    Update: FAKE keywords list has been further expanded.
    Default classification is set to 'REAL' if no strong keywords are detected.
    """
    text = text.lower()
    
    # Enhanced keyword lists
    # Keywords for Fake News (Sensational, Health, Financial scams, Conspiracies)
    fake_keywords = ['cure', 'miracle', 'secret', 'overnight', 'aliens', 'scandal', 'shocking', 'hoax', 'exclusive', 'breaking now', 'unbelievable', 'warning', 'danger', 'cancer', 'radiation', 'exposed', 'urgent', 'virus spread', 'no one is telling you', 'demise', 'secretly', 'ban', 'health threat', 'faked', 'fake', 'conspiracy', 'money', 'free', 'win', 'click here', 'breaking news', 'massive', 'unusual', 'proof']
    
    # Keywords for Real News (Official, Scientific, Governmental, Established Facts)
    real_keywords = ['study', 'report', 'announce', 'research', 'government', 'scientists', 'official', 'confirmed', 'evidence', 'budget', 'congress', 'minister', 'spokesperson', 'approved', 'investigation', 'quarterly', 'policy', 'economy', 'court', 'military']

    # Check for FAKE keywords
    if any(word in text for word in fake_keywords):
        return "FAKE"
    
    # Check for REAL keywords
    elif any(word in text for word in real_keywords):
        return "REAL"
        
    else:
        # If no strong keywords are found, it defaults to REAL. 
        # Since most text in a general setting tends to be non-sensational/real, we keep this default.
        return "REAL"

# -------------------------
# Prediction Function
# -------------------------
def predict_text(text):
    """Performs the main ML prediction, falling back to rule-based if needed."""
    if model is None or w2v_model is None:
        return "Error: Models failed to load."

    review = preprocess_text(text)

    if len(review) == 0:
        return "Please enter valid text"

    # Known words from Word2Vec
    if hasattr(w2v_model, 'wv'):
        known_words = [word for word in review if word in w2v_model.wv.key_to_index]
    else:
        return keyword_fallback(text) 


    # If no known words, use keyword fallback
    if len(known_words) == 0:
        print("Falling back to keyword classification.")
        return keyword_fallback(text)

    # Vectorization
    try:
        vector_size = w2v_model.vector_size if hasattr(w2v_model, 'vector_size') else 100
        
        review_vect = np.array([w2v_model.wv[word] for word in known_words])
        review_vect = np.mean(review_vect, axis=0)

        # Fix to 100 dimensions (assuming the model was trained on 100-dim vectors)
        if review_vect.shape[0] > vector_size:
            review_vect = review_vect[:vector_size]
        elif review_vect.shape[0] < vector_size:
            review_vect = np.pad(review_vect, (0, vector_size - review_vect.shape[0]), 'constant')

        review_vect = review_vect.reshape(1, -1)

        # Model prediction
        prediction = model.predict(review_vect)

        # Simplified output to just FAKE or REAL
        if prediction[0] == 0:
            return "FAKE"
        else:
            return "REAL"
            
    except Exception as e:
        print(f"ML Prediction failed: {e}. Falling back to keyword classification.")
        return keyword_fallback(text)


# -------------------------
# Routes (Unchanged)
# -------------------------
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def webapp():
    try:
        text = request.form.get('text', '').strip()
        if text == "":
            result = "Please enter some text!"
        else:
            result = predict_text(text)
        return render_template('index.html', text=text, result=result)
    except Exception as e:
        return render_template('index.html', text=request.form.get('text', ''), result=f"Prediction Error: {e}")

@app.route('/predict/', methods=['GET', 'POST'])
def api():
    try:
        text = request.args.get("text") or request.json.get("text")
        if not text:
            return jsonify(error="No text provided"), 400
        prediction = predict_text(text)
        return jsonify(prediction=prediction)
    except Exception as e:
        return jsonify(error=str(e)), 500

# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
