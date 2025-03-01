# Spotify Reviews Chatbot

A specialized Q&A chatbot for analyzing Spotify app reviews from the Google Play Store, built with Streamlit, FAISS, and Hugging Face models.

## Overview

This chatbot allows users to ask questions about Spotify app reviews and get AI-powered insights. It can analyze sentiment, identify trends, summarize feedback, and answer specific questions based on the review dataset.

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

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Hugging Face API token:
   ```
   HF_TOKEN=your_token_here
   ```

## Dataset

Place your Spotify reviews dataset in the `dataset` folder as `SPOTIFY_REVIEWS.csv`. The dataset should have the following columns:
- review_id
- review_text
- review_rating
- review_likes
- author_app_version
- review_timestamp

## Usage

Run the application:
```
streamlit run chatbot.py
```

### Example Queries

- "What are users saying about the latest version?"
- "Summarize the main complaints in reviews from the past month"
- "What's the average rating for version 8.7.18.459?"
- "How has user sentiment changed over time?"
- "What features do users request most often?"

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

## Improvements and Extensions

- **Data Visualization**: Add charts and graphs for trend analysis
- **Feedback Loop**: Implement user feedback on response quality
- **More Analysis Tools**: Add comparison between versions, feature-focused analysis
- **Multi-platform**: Extend to analyze reviews from App Store and other platforms
- **Custom Training**: Fine-tune the language model on Spotify-specific terminology

## License

[Your license information]

## Contact

[Your contact information]
