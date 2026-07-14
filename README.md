# VoyageAI: Intelligent Travel Recommendation System

### VoyageAI – Intelligent Travel Recommendation System
*Python, Flask, NumPy, Pandas, Scikit-Learn, HTML5, CSS3, JavaScript*
- Developed an end-to-end travel recommendation platform generating personalized day-by-day itineraries and hotel matches based on dynamic user preferences, category ratings, and budgets.
- Built a NumPy-based Restricted Boltzmann Machine (RBM) neural network model to learn hidden user preferences from 33,000+ review ratings, forecasting rating scores for unseen attractions.
- Implemented a custom SGD-based Matrix Factorization collaborative filtering algorithm (SimpleMF) to recommend top-rated hotels matching user-preferred amenities.
- Designed a premium Single Page Application (SPA) web interface using a dark glassmorphic layout, Outfit typography, glowing hover transitions, and responsive grid layouts.
- Engineered a Flask REST API backend with modular service architecture, on-the-fly model training fallbacks, and optimized model batch parameters, achieving real-time query latencies under 0.1 seconds.

---

## 🛠️ Technology Stack

* **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism & HSL Color System), Vanilla JavaScript (ES6+), FontAwesome Icons.
* **Backend**: Flask (Python REST API), Flask-CORS.
* **Algorithms/Data Models**: NumPy, Pandas, Scikit-learn (MinMax Scaling), Matplotlib (plots & diagnostics).

---

## 📂 Project Structure

```
├── backend/            # Python core recommendation engine & Flask Server
│   ├── app.py              # Flask server and REST API controller
│   ├── rbm.py              # Restricted Boltzmann Machine implementation (NumPy)
│   ├── utils.py            # Preprocessing, data clean, and free energy helper functions
│   ├── attractions_recc.py # Attractions filtering and nearest-neighbor clustering
│   └── hotel_recc.py       # Custom Matrix Factorization hotel matching logic
├── frontend/           # Single Page Application assets
│   ├── index.html          # Web UI layout
│   ├── style.css           # Glassmorphism and responsive grid styles
│   └── app.js              # State management and AJAX API integration
├── data/               # Curated JSON datasets (attractions, reviews, amenities)
│   ├── etl/                # Core rating data tables
│   └── outputs/            # Supplemental categories URLs mapping
├── models/             # Saved model outputs (weights and serialized model matrices)
├── notebooks/          # Exploratory Data Analysis & training Jupyter Notebooks
└── downloads/          # Local collection of travel destination images
```

---

## ⚙️ Setup & Installation

### 1. Prerequisites
Make sure Python 3.8+ is installed on your computer.

### 2. Install Required Python Libraries
Install the core requirements:
```bash
pip install flask flask-cors pandas numpy scikit-learn matplotlib
```

### 3. Run the Web Server
Navigate to the repository folder and start the server:
```bash
python backend/app.py
```
*Note: Debug mode is off by default for stability on Windows systems.*

### 4. Open in Your Browser
Open your browser and navigate to:
👉 **[http://localhost:5000](http://localhost:5000)**

---

## 🧠 How the Algorithms Work

### 1. Attraction Recommendation (Restricted Boltzmann Machine)
* User ratings for attraction categories are normalized.
* The preprocessor constructs a sparse interaction vector mapped to unique attractions.
* An RBM with $V$ visible units (matching attraction counts) and $H$ hidden units (representing learned features) runs Gibb's sampling and updates model weights via Contrastive Divergence (CD).
* If a pre-trained model folder is missing, the backend automatically trains a lightweight version on the fly under 0.1 seconds, predicting candidate scores for unvisited locations.
* Reconstructed ratings are scaled (MinMax 0-5) and mapped back to details inside the database.

### 2. Hotel Collaborative Filtering (SimpleMF Matrix Factorization)
* A binary match is run comparing user amenity checkboxes against the amenities array of TripAdvisor hotels.
* Matching length is mapped to a candidate rating (1 to 5).
* A custom NumPy-based Matrix Factorization model (ALS/SGD updates) splits data 80/20, evaluates candidates, selects the best rank dimensions ($K \in [4, 8, 12]$), and generates predicted ratings across the hotel database, returning the top matches.

---

## 👤 Author

* **Lakshya Sahu**
  * **Email**: [thelakshyasahu@gmail.com](mailto:thelakshyasahu@gmail.com)
  * **GitHub**: [@Lakshya-Sahu47](https://github.com/Lakshya-Sahu47)
  * **LinkedIn**: [Lakshya Sahu](https://www.linkedin.com/in/lakshya-sahu-9a60679173) *(Link based on profile)*

