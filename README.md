# movie-review-sentiment-analysis
Movie Review Sentiment Analyzer uses ML to classify movie reviews as positive or negative. It preprocesses text data, trains a Logistic Regression model with TF-IDF features, and features a Streamlit web app for real-time sentiment prediction. A beginner-friendly NLP and ML deployment project.
## Features

- Text cleaning and preprocessing using NLTK
- TF-IDF vectorization for feature extraction
- Logistic Regression model for sentiment classification
- Interactive web app built with Streamlit for real-time predictions

# steps:

1.Clone the repository:
   
   git clone https://github.com/yourusername/movie-review-sentiment-analyzer.git
   cd movie-review-sentiment-analyzer


2.Install dependencies:

   pip install -r requirements.txt
   
3.Download and prepare dataset:

  Download the IMDB Large Movie Review Dataset
  Extract and place the aclImdb folder inside the project directory.

4.Preprocess the data:

  python preprocess.py

5.Train the model:

  python train.py

6.Run the Streamlit app:

  python -m streamlit run app.py

## Usage

- Enter a movie review in the app text box.

- Click "Predict Sentiment" to see if the review is positive or negative.

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- nltk
- streamlit
- matplotlib
- wordcloud

