<img width="226" alt="image" src="https://github.com/user-attachments/assets/caf308dd-b83a-4d5a-a882-55d89729001d" />

# Spotify Reviews Q&A Chatbot

A specialized Q&A chatbot for analyzing Spotify app reviews from the Google Play Store, built with Streamlit, FAISS, and HuggingFace models.

## Overview

This chatbot allows users to ask questions about Spotify Google reviews and get AI-powered insights. It can analyze sentiment, identify trends, summarize feedback, and answer specific questions based on the review dataset.

## Features

- **Natural language Q&A**: Ask questions about Spotify reviews in plain English
- **Semantic search**: Identifies relevant reviews using FAISS similarity search
- **Multiple analysis modes**:
  - Question answering about specific aspects of reviews
  - Summarization of review content
  - Trend analysis over time 
  - Basic calculations
- **Context-aware**: Maintains conversation history for follow-up questions
- **Detailed insights**: Shows retrieved reviews, quality scores and thinking process
- **Interactive UI**: Built with Streamlit for easy user interaction

## Requirements

- Python 3.8+
- Streamlit
- FAISS
- SentenceTransformer
- HuggingFace Inference API
- Pandas
- NumPy
- Tiktoken
- python-dotenv

## Installation & Usage

1. Download chatbot.py and REQUIREMENTS.txt
2. Download all files from this [Kaggle repository](https://www.kaggle.com/datasets/ngchunyiu/spotify-google-reviews-dataset/data)
3. Create a `.env` file with your [HuggingFace API token](https://huggingface.co/settings/tokens):
   ```
   HF_TOKEN=your_huggingface_token_here
   ```
4. Place the following files into your working directory:
   ```
   chatbot.py
   SPOTIFY_REVIEWS.csv
   embeddings.npy
   faiss_index.index
   REQUIREMENTS.txt
   .env
   ```
6. Install dependencies:
   ```
   pip install -r REQUIREMENTS.txt
   ```
7. Run the application:
   ```
   streamlit run chatbot.py
   ```

## Dataset

The CSV file contains 3.4+ million Spotify Google Reviews with the following columns:
- review_id
- review_text
- review_rating
- review_likes
- author_app_version
- review_timestamp

## Example Query
<img width="1505" alt="image" src="https://github.com/user-attachments/assets/ea3e6354-72b5-4989-96b1-689422ef5edb" />
<img width="1498" alt="image" src="https://github.com/user-attachments/assets/5f695cf9-13a0-4278-8b12-3714738e43c7" />
<img width="1499" alt="image" src="https://github.com/user-attachments/assets/f319c7c0-b5e9-4e52-8d8b-3775ae58a2a4" />
<img width="1497" alt="image" src="https://github.com/user-attachments/assets/9d0ec03d-fc93-427b-a76f-77295bb8776b" />
Within the dropdowns:
<img width="1494" alt="image" src="https://github.com/user-attachments/assets/2281582c-93f9-489b-a48d-4eb425ab1beb" />
<img width="1498" alt="image" src="https://github.com/user-attachments/assets/17ff34e4-5180-440e-ae32-1dbda57646e2" />
<img width="1498" alt="image" src="https://github.com/user-attachments/assets/66aa402d-e9ec-437c-8994-edee0ac55428" />
<img width="1499" alt="image" src="https://github.com/user-attachments/assets/ca6ba57b-8eac-42ff-878b-779d89cfd694" />
<img width="1497" alt="image" src="https://github.com/user-attachments/assets/6ec4ffdf-92a9-48e3-97af-e63f4e841d82" />
<img width="1497" alt="image" src="https://github.com/user-attachments/assets/e6d931e0-3a26-4f43-b9f9-9c9b61b911c1" />
<img width="1498" alt="image" src="https://github.com/user-attachments/assets/e8fe8873-1804-4a04-aaeb-9ab7f8c85efd" />
<img width="1500" alt="image" src="https://github.com/user-attachments/assets/0abc438e-08d1-4c90-a868-6e84b2faceec" />
<img width="1497" alt="image" src="https://github.com/user-attachments/assets/82569e99-bf62-4c20-8b21-9e9ebcd41f2a" />
<img width="1501" alt="image" src="https://github.com/user-attachments/assets/43c1bbcb-0f13-4c65-9d98-73d3881b0d36" />
<img width="1499" alt="image" src="https://github.com/user-attachments/assets/cbb20197-d2e1-434c-a665-512ad778a2fa" />
<img width="1499" alt="image" src="https://github.com/user-attachments/assets/844d2752-cd7d-44e5-9391-7ca2291f6fd6" />
<img width="1498" alt="image" src="https://github.com/user-attachments/assets/0b640db2-b528-4426-a98d-f9f2c3bd8e18" />
<img width="1502" alt="image" src="https://github.com/user-attachments/assets/a2c405d1-fa13-4b9d-ab44-c0c5effe7f9a" />


## Technical Details

### Components

1. **Embedding & Retrieval System**:
   - Uses SentenceTransformer for generating embeddings
   - FAISS index for efficient similarity search
   - Optimized for batched processing of large review datasets

2. **Context Management**:
   - Maintains conversation history
   - Detects follow-up questions
   - Extracts referenced entities (versions, time frames)

3. **Response Generation**:
   - Uses DeepSeek-R1-Distill-Qwen-32B via Hugging Face Inference API
   - Customized prompting for different analysis types
   - Token estimation to avoid prompt truncation

4. **Quality Evaluation**:
   - Calculates quality scores based on:
     - Cosine similarity between question and answer
     - Keyword overlap with retrieved reviews

### Processing Flow

1. Query analysis to determine intent
2. Context extraction for follow-up questions
3. Selection of appropriate tool (QA, summarization, trends, calculation)
4. Filtering of reviews by version and time frame
5. Semantic search for relevant reviews
6. Prompt construction with retrieved context
7. LLM-based response generation
8. Quality evaluation

## Customization

- Modify `AGENT_PROMPT` to change the assistant's personality and style
- Adjust `CHUNK_SIZE` and `MAX_TOKENS` for different review dataset sizes
- Change the embedding model by replacing `paraphrase-MiniLM-L6-v2` with another SentenceTransformer model

## Potential Improvements and Extensions

- **Data Visualization**: Add charts and graphs for trend analysis
- **Feedback Loop**: Implement user feedback on response quality
- **More Analysis Tools**: Add comparison between versions, feature-focused analysis
- **Multi-platform**: Extend to analyze reviews from App Store and other platforms
- **Custom Training**: Fine-tune the language model on Spotify-specific terminology

## Video Demo
https://github.com/user-attachments/assets/272551f9-f1be-4c5a-b087-670cdcbbe8b5
