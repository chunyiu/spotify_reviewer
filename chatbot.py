import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import streamlit as st
import re
import dateutil.relativedelta as relativedelta
import asyncio
import time
from typing import Optional, List
import traceback
import tiktoken
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face API token not found. Please set HF_TOKEN in the .env file.")

# Initialize Hugging Face Inference Client
client = InferenceClient(token=hf_token)

# Constants
DATA_PATH = "SPOTIFY_REVIEWS.csv"
INDEX_PATH = "faiss_index.index"
EMBEDDINGS_PATH = "embeddings.npy"
CHUNK_SIZE = 10000
MAX_TOKENS = 2048
CONTEXT_WINDOW = 3  # Number of previous exchanges to consider for context
AGENT_PROMPT = """
You are Spotify Insights, an advanced analytics agent developed by Spotify to help analyze user reviews from the Google Play Store. 

Your primary responsibilities is:
1. Answer questions about Spotify app reviews accurately from the perspective of an agent built by Spotify

Guidelines:
- When specific information isn't available in the reviews, clearly state this limitation
- Maintain a professional, helpful tone representing Spotify's commitment to user feedback
- Format insights clearly with appropriate spacing and structure. Use bullet points when necessary
- Be specific in your response. Respond clearly with adequate elaboration
- When analyzing version-specific issues, clearly indicate which app version is being discussed

Remember that you are an official Spotify tool designed to provide data-driven insights to help the team understand and improve the user experience.
"""

# Load the entire DataFrame
df = pd.read_csv(DATA_PATH, usecols=["review_id", "review_text", "review_rating", "review_likes", "author_app_version", "review_timestamp"])
df['review_timestamp'] = pd.to_datetime(df['review_timestamp'])
reviews = df['review_text'].dropna().tolist()

# Compute embeddings in batches
def compute_embeddings(reviews, embedder, batch_size=1000):
    embeddings = []
    for i in range(0, len(reviews), batch_size):
        batch = reviews[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} of {len(reviews) // batch_size + 1}")
        batch_embeddings = embedder.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings)

# Create and save Faiss index
def create_and_save_index(embeddings, index_path):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_path)
    return index

# Load or compute embeddings and index
def load_or_compute_index(reviews, embedder):
    if os.path.exists(INDEX_PATH) and os.path.exists(EMBEDDINGS_PATH):
        index = faiss.read_index(INDEX_PATH)
        embeddings = np.load(EMBEDDINGS_PATH)
    else:
        embeddings = compute_embeddings(reviews, embedder)
        index = create_and_save_index(embeddings, INDEX_PATH)
        np.save(EMBEDDINGS_PATH, embeddings)
    return index, embeddings

def estimate_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# Thinking logger
class ThinkingLogger:
    def __init__(self):
        self.thoughts = []
        self.start_time = None
        
    def start(self):
        self.start_time = time.time()
        self.thoughts = []
        
    def log(self, thought):
        elapsed = time.time() - self.start_time
        self.thoughts.append(f"[{elapsed:.2f}s] {thought}")
        
    def get_thoughts(self):
        return self.thoughts
        
    def get_elapsed_time(self):
        if self.start_time is None:
            return 0
        return time.time() - self.start_time

# Function to call Hugging Face API
async def generate_response(prompt, temperature=0.5, max_tokens=2048, thinking_logger=None):
    try:
        if thinking_logger:
            thinking_logger.log(f"Generating response for prompt: {prompt[:100]}...")
            
        # Add the agent prompt to the beginning of the user's prompt
        full_prompt = AGENT_PROMPT + "\n\n" + prompt
        
        messages = [{"role": "user", "content": full_prompt}]
        response = ""
        
        if thinking_logger:
            thinking_logger.log("Making API call to DeepSeek-R1-Distill-Qwen-32B")
            
        stream = client.chat_completion(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.7,
            stream=True
        )
        
        if thinking_logger:
            thinking_logger.log("Stream created successfully, collecting chunks...")
        
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response += content
        
        if thinking_logger:
            thinking_logger.log(f"Response generated ({len(response)} chars)")
        
        return response
    except Exception as e:
        if thinking_logger:
            thinking_logger.log(f"Error in API call: {str(e)}")
            
        print(f"Error in API call: {str(e)}")
        print(traceback.format_exc())
        return f"I encountered an error while processing your request: {str(e)}"

# Context handling functions
def get_conversation_context(history, current_query, thinking_logger=None):
    """Extract context from conversation history"""
    if thinking_logger:
        thinking_logger.log("Retrieving conversation context")
    
    # Get last CONTEXT_WINDOW exchanges
    relevant_history = history[-CONTEXT_WINDOW*2:] if len(history) > CONTEXT_WINDOW*2 else history
    
    context_str = ""
    for i in range(0, len(relevant_history), 2):
        if i+1 < len(relevant_history):
            context_str += f"User: {relevant_history[i]['content']}\n"
            context_str += f"Assistant: {relevant_history[i+1]['content']}\n\n"
    
    context_str += f"Current User Query: {current_query}\n"

    if thinking_logger:
        thinking_logger.log(f"Retrieved {len(relevant_history)//2} previous exchanges for context")
        
    return context_str

def is_follow_up_query(query, thinking_logger=None):
    """Determine if query is likely a follow-up question"""
    if thinking_logger:
        thinking_logger.log("Checking if query is a follow-up")
        
    # Check for common follow-up indicators
    follow_up_indicators = [
        r"\b(what|how|why) about\b",
        r"\b(can|could) you\b",
        r"\band\b.*\?",
        r"\balso\b",
        r"\bthen\b",
        r"\banother\b",
        r"^(what|why|how|when|where|who|which)",
        r"\bthat\b",
        r"\bthose\b",
        r"\bthey\b",
        r"\bit\b"
    ]
    
    is_follow_up = False
    for pattern in follow_up_indicators:
        if re.search(pattern, query.lower()):
            is_follow_up = True
            if thinking_logger:
                thinking_logger.log(f"Detected follow-up indicator: {pattern}")
            break
    
    if thinking_logger:
        thinking_logger.log(f"Query determined to be {'a follow-up' if is_follow_up else 'a new question'}")
        
    return is_follow_up

def get_referenced_entities(history, thinking_logger=None):
    """Extract potential entities that might be referenced in follow-ups"""
    if thinking_logger:
        thinking_logger.log("Extracting referenced entities from previous exchanges")
    
    if not history:
        return []
        
    entities = []
    
    # Check last exchange for key entities
    if len(history) >= 2:
        last_query = history[-2]['content'].lower()
        last_response = history[-1]['content'].lower()
        
        # Extract version mentions
        version_matches = re.findall(r'version (\d+\.\d+\.\d+)', last_query + " " + last_response)
        entities.extend([f"version {v}" for v in version_matches])
        
        # Extract time frame mentions
        time_patterns = [
            (r'last (\d+) days', 'last {} days'),
            (r'last month', 'last month'),
            (r'past year', 'past year'),
            (r'in (\d{4})', 'in {}')
        ]
        
        for pattern, template in time_patterns:
            matches = re.findall(pattern, last_query + " " + last_response)
            for match in matches:
                if isinstance(match, tuple):
                    entities.append(template.format(*match))
                else:
                    entities.append(template.format(match))
    
    if thinking_logger:
        thinking_logger.log(f"Extracted entities: {entities}")
        
    return entities

# Data handling functions
def extract_version(query, df, thinking_logger=None):
    if thinking_logger:
        thinking_logger.log("Extracting version information from query")
        
    query_lower = query.lower()
    if "latest version" in query_lower:
        latest_version = df.loc[df['review_timestamp'].idxmax(), 'author_app_version']
        if thinking_logger:
            thinking_logger.log(f"Latest version identified: {latest_version}")
        return latest_version
    match = re.search(r'version (\d+\.\d+\.\d+)', query)
    result = match.group(1) if match else None
    
    if thinking_logger:
        thinking_logger.log(f"Version extracted: {result}")
        
    return result

def extract_time_frame(query, thinking_logger=None):
    if thinking_logger:
        thinking_logger.log("Extracting time frame from query")
        
    query_lower = query.lower()
    result = None
    
    if "recent" in query_lower or "last 30 days" in query_lower:
        result = ("last", 30, "days")
    elif "last month" in query_lower:
        result = ("last", 1, "month")
    elif "past year" in query_lower:
        result = ("last", 1, "year")
    elif re.search(r'in (\d{4})', query_lower):
        year = int(re.search(r'in (\d{4})', query_lower).group(1))
        result = ("year", year)
        
    if thinking_logger:
        thinking_logger.log(f"Time frame identified: {result}")
        
    return result

def filter_by_time(df, time_frame, thinking_logger=None):
    if time_frame is None:
        return df
        
    if thinking_logger:
        thinking_logger.log(f"Filtering by time frame: {time_frame}")
        
    max_timestamp = df['review_timestamp'].max()
    filtered_df = df
    
    if time_frame[0] == "last":
        if time_frame[2] == "days":
            delta = pd.Timedelta(days=time_frame[1])
        elif time_frame[2] == "month":
            delta = relativedelta.relativedelta(months=time_frame[1])
        elif time_frame[2] == "year":
            delta = relativedelta.relativedelta(years=time_frame[1])
        cutoff = max_timestamp - delta
        filtered_df = df[df['review_timestamp'] >= cutoff]
    elif time_frame[0] == "year":
        year = time_frame[1]
        filtered_df = df[df['review_timestamp'].dt.year == year]
        
    if thinking_logger:
        thinking_logger.log(f"Filtered to {len(filtered_df)} reviews")
        
    return filtered_df

# Tool functions
async def answer_question(query, embedder, index, df, k=10, thinking_logger=None, context=None):
    try:
        if thinking_logger:
            thinking_logger.log("Using question answering tool")
            
        # Include context information in the query if it's a follow-up
        if context:
            thinking_logger.log("Processing follow-up query with context")
            enriched_query = f"{context}\nCurrent query: {query}"
            thinking_logger.log(f"Enriched query: {enriched_query[:100]}...")
        else:
            enriched_query = query
            
        version = extract_version(enriched_query, df, thinking_logger)
        time_frame = extract_time_frame(enriched_query, thinking_logger)
        filtered_df = filter_by_time(df, time_frame, thinking_logger) if time_frame else df
        
        if version:
            filtered_df = filtered_df[filtered_df['author_app_version'] == version]
            if thinking_logger:
                thinking_logger.log(f"Filtered to version {version}: {len(filtered_df)} reviews")

        query_lower = enriched_query.lower()
        if re.search(r'(average|mean) rating', query_lower):
            result = f"The average rating is {filtered_df['review_rating'].mean():.2f}."
            if thinking_logger:
                thinking_logger.log(f"Calculated average rating: {result}")
            return result, []
            
        elif re.search(r'how many reviews', query_lower):
            result = f"There are {len(filtered_df)} reviews."
            if thinking_logger:
                thinking_logger.log(result)
            return result, []

        if thinking_logger:
            thinking_logger.log("Computing question embedding for semantic search")
            
        question_embedding = embedder.encode([query], convert_to_tensor=True)
        distances, indices = index.search(question_embedding.cpu().numpy(), 1000)
        
        if thinking_logger:
            thinking_logger.log("Filtering search results to match time/version criteria")
            
        filtered_indices = [i for i in indices[0] if i in set(filtered_df.index)]
        top_k_indices = filtered_indices[:k]

        if not top_k_indices:
            if thinking_logger:
                thinking_logger.log("No relevant reviews found")
            return "No relevant reviews found.", []

        if thinking_logger:
            thinking_logger.log(f"Found {len(top_k_indices)} relevant reviews")
            
        retrieved_reviews = [
            f"Review (Rating: {df.iloc[idx]['review_rating']}, "
            f"Likes: {df.iloc[idx]['review_likes']}, "
            f"Version: {df.iloc[idx]['author_app_version']}, "
            f"Date: {df.iloc[idx]['review_timestamp'].date()}): "
            f"{df.iloc[idx]['review_text']}"
            for idx in top_k_indices
        ]
        
        # Construct prompt with context if available
        if context:
            prompt = f"Conversation history:\n{context}\n\nCurrent Question: {query}\n\nRelevant Reviews:\n"
        else:
            prompt = f"Question: {query}\n\nRelevant Reviews:\n"
            
        prompt += "\n".join(f"- {r}" for r in retrieved_reviews)
        
        if thinking_logger:
            thinking_logger.log("Building prompt with retrieved reviews")
            thinking_logger.log(f"Estimated token count: {estimate_tokens(prompt)}")
            
        if estimate_tokens(prompt) > MAX_TOKENS - 500:
            if thinking_logger:
                thinking_logger.log("Prompt too long, using shorter summaries")
            
            # More aggressive trimming for prompts with context
            max_reviews = 7 if context else 10
            top_k_indices = top_k_indices[:max_reviews]
            
            if context:
                prompt = f"Conversation history:\n{context}\n\nCurrent Question: {query}\n\nSummaries:\n"
            else:
                prompt = f"Question: {query}\n\nSummaries:\n"
                
            prompt += "\n".join(f"- {df.iloc[idx]['review_text'][:150]}..." for idx in top_k_indices)
        
        # Add instructions for handling follow-up queries
        if context:
            prompt += "\n\nPlease provide a concise and accurate answer based only on the provided reviews. Ensure your answer directly addresses the current question while maintaining context from the conversation history. If the information is not in the reviews, state that clearly."
        else:
            prompt += "\n\nPlease provide a concise and accurate answer based only on the provided reviews. If the information is not in the reviews, state that clearly."

        if thinking_logger:
            thinking_logger.log("Sending prompt to language model")
            
        response = await generate_response(prompt, thinking_logger=thinking_logger)
        
        return response, retrieved_reviews
    except Exception as e:
        if thinking_logger:
            thinking_logger.log(f"Error: {str(e)}")
        return f"Error processing query: {str(e)}", []

async def summarize_reviews(query, df, thinking_logger=None, context=None):
    try:
        if thinking_logger:
            thinking_logger.log("Using review summarization tool")
            
        # Include context information in the query if it's a follow-up
        if context:
            thinking_logger.log("Processing follow-up summarization request with context")
            enriched_query = f"{context}\nCurrent query: {query}"
            thinking_logger.log(f"Enriched query: {enriched_query[:100]}...")
        else:
            enriched_query = query
            
        version = extract_version(enriched_query, df, thinking_logger)
        time_frame = extract_time_frame(enriched_query, thinking_logger)
        filtered_df = filter_by_time(df, time_frame, thinking_logger) if time_frame else df
        
        if version:
            filtered_df = filtered_df[filtered_df['author_app_version'] == version]
            if thinking_logger:
                thinking_logger.log(f"Filtered to version {version}: {len(filtered_df)} reviews")
        
        if filtered_df.empty:
            if thinking_logger:
                thinking_logger.log("No reviews found matching criteria")
            return "No reviews found matching the criteria", []
            
        # Take a random sample if there are too many reviews
        if len(filtered_df) > 50:
            if thinking_logger:
                thinking_logger.log(f"Sampling 50 reviews from {len(filtered_df)} total")
            filtered_df = filtered_df.sample(50)
            
        reviews_text = filtered_df['review_text'].tolist()
        
        if thinking_logger:
            thinking_logger.log(f"Preparing summary prompt with {len(reviews_text)} reviews")
        
        # Construct prompt with context if available
        if context:
            prompt = f"Conversation history:\n{context}\n\nCurrent request: Summarize these Spotify app reviews and address: {query}\n\n"
        else:    
            prompt = f"Summarize these Spotify app reviews:\n\n"
            
        prompt += "\n".join([f"- {r}" for r in reviews_text])
        
        if context:
            prompt += f"\n\nProvide a concise summary highlighting the main themes, positive points, and areas of concern. Address the current request while maintaining context from previous exchanges."
        else:
            prompt += f"\n\nAdditional instructions: {query}\n\nProvide a concise summary highlighting the main themes, positive points, and areas of concern."
        
        if thinking_logger:
            thinking_logger.log("Sending summary prompt to language model")
            
        response = await generate_response(prompt, thinking_logger=thinking_logger)
        return response, reviews_text
    except Exception as e:
        if thinking_logger:
            thinking_logger.log(f"Error: {str(e)}")
        return f"Error summarizing reviews: {str(e)}", []

async def analyze_trends(query, df, thinking_logger=None, context=None):
    try:
        if thinking_logger:
            thinking_logger.log("Using trend analysis tool")
            
        # Include context information in the query if it's a follow-up
        if context:
            thinking_logger.log("Processing follow-up trend analysis with context")
            enriched_query = f"{context}\nCurrent query: {query}"
            thinking_logger.log(f"Enriched query: {enriched_query[:100]}...")
        else:
            enriched_query = query
            
        time_frame = extract_time_frame(enriched_query, thinking_logger)
        filtered_df = filter_by_time(df, time_frame, thinking_logger) if time_frame else df
        
        if filtered_df.empty:
            if thinking_logger:
                thinking_logger.log("No reviews found matching criteria")
            return "No reviews found matching the criteria", []
            
        if thinking_logger:
            thinking_logger.log("Calculating monthly averages and counts")
            
        filtered_df['review_month'] = filtered_df['review_timestamp'].dt.to_period('M')
        avg_rating = filtered_df.groupby('review_month')['review_rating'].mean()
        monthly_counts = filtered_df.groupby('review_month').size()
        
        trend_data = "Average rating per month:\n" + "\n".join(
            f"{month}: {rating:.2f} (from {monthly_counts[month]} reviews)" 
            for month, rating in avg_rating.items()
        )
        
        if thinking_logger:
            thinking_logger.log(f"Prepared trend data for {len(avg_rating)} months")
        
        # Construct prompt with context if available
        if context:
            prompt = f"Conversation history:\n{context}\n\nCurrent Question: {query}\n\nTrend Data:\n{trend_data}\n\n"
            prompt += "Please analyze these trends and provide insights, taking into account the conversation history and focusing on the current question."
        else:
            prompt = f"Question: {query}\n\nTrend Data:\n{trend_data}\n\nPlease analyze these trends and provide insights."
        
        if thinking_logger:
            thinking_logger.log("Sending trend analysis prompt to language model")
            
        response = await generate_response(prompt, thinking_logger=thinking_logger)
        return response, []
    except Exception as e:
        if thinking_logger:
            thinking_logger.log(f"Error: {str(e)}")
        return f"Error analyzing trends: {str(e)}", []

async def perform_calculation(query, thinking_logger=None, context=None):
    try:
        if thinking_logger:
            thinking_logger.log("Using calculation tool")
            
        # For calculations, context is rarely needed but we include it in logs
        if context:
            thinking_logger.log("Processing calculation with conversation context")
        
        # Extract the mathematical expression from the query
        math_expr = re.search(r'([\d\+\-\*\/\(\)\.\s]+)', query)
        if math_expr:
            expr = math_expr.group(1).strip()
            if thinking_logger:
                thinking_logger.log(f"Extracted expression: {expr}")
            result = eval(expr, {"__builtins__": {}}, {"np": np})
            if thinking_logger:
                thinking_logger.log(f"Calculated result: {result}")
            return f"Result of {expr} = {result}", []
        else:
            if thinking_logger:
                thinking_logger.log("No valid mathematical expression found")
            return "Could not find a valid mathematical expression in your query.", []
    except Exception as e:
        if thinking_logger:
            thinking_logger.log(f"Error: {str(e)}")
        return f"Error calculating: {str(e)}", []

# Quality Score
def quality_score(question: str, response: str, retrieved_reviews: Optional[List[str]], thinking_logger=None) -> float:
    if not retrieved_reviews:
        return 0
    try:
        if thinking_logger:
            thinking_logger.log("Calculating quality score")
            
        embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        question_emb = embedder.encode([question])[0]
        response_emb = embedder.encode([response])[0]
        cosine_sim = cosine_similarity([question_emb], [response_emb])[0][0]

        if thinking_logger:
            thinking_logger.log(f"Cosine similarity: {cosine_sim:.2f}")

        response_keywords = set(re.findall(r'\w+', response.lower()))
        reviews_text = ' '.join(retrieved_reviews).lower()
        overlap = sum(1 for word in response_keywords if word in reviews_text)
        overlap_score = overlap / len(response_keywords) if response_keywords else 0

        if thinking_logger:
            thinking_logger.log(f"Keyword overlap score: {overlap_score:.2f}")

        final_score = (cosine_sim + overlap_score) / 2
        
        if thinking_logger:
            thinking_logger.log(f"Final quality score: {final_score:.2f}")
            
        return final_score
    except Exception as e:
        if thinking_logger:
            thinking_logger.log(f"Error calculating quality score: {e}")
        print(f"Error calculating quality score: {e}")
        return 0

# Determine which tool to use based on the query
async def process_query(query: str, embedder, index, df, history=None) -> tuple[str, Optional[List[str]], Optional[float], List[str], float]:
    thinking_logger = ThinkingLogger()
    thinking_logger.start()
    
    query_lower = query.lower()
    thinking_logger.log(f"Processing query: {query}")
    
    # Check if this is a follow-up question
    context = None
    if history and len(history) > 1:
        is_follow_up = is_follow_up_query(query, thinking_logger)
        
        if is_follow_up:
            context = get_conversation_context(history, query, thinking_logger)
            thinking_logger.log("This appears to be a follow-up question")
            thinking_logger.log(f"Context length: {len(context)} characters")
    
    try:
        # Look for specific query patterns
        if any(kw in query_lower for kw in ["calculate", "what is", "compute", "equals", "equal to"]) and re.search(r'[\d\+\-\*\/]', query_lower):
            thinking_logger.log("Detected calculation query")
            response, reviews = await perform_calculation(query, thinking_logger, context)
            elapsed_time = thinking_logger.get_elapsed_time()
            return response, reviews, None, thinking_logger.get_thoughts(), elapsed_time
            
        elif any(kw in query_lower for kw in ["summarize", "summary", "overall sentiment", "key themes", "main points"]):
            thinking_logger.log("Detected summarization query")
            response, reviews = await summarize_reviews(query, df, thinking_logger, context)
            score = quality_score(query, response, reviews, thinking_logger)
            elapsed_time = thinking_logger.get_elapsed_time()
            return response, reviews, score, thinking_logger.get_thoughts(), elapsed_time
            
        elif any(kw in query_lower for kw in ["trend", "over time", "change in", "changing", "monthly", "yearly"]):
            thinking_logger.log("Detected trend analysis query")
            response, reviews = await analyze_trends(query, df, thinking_logger, context)
            elapsed_time = thinking_logger.get_elapsed_time()
            return response, reviews, None, thinking_logger.get_thoughts(), elapsed_time
            
        else:
            # Default to question answering
            thinking_logger.log("Using default question answering")
            response, reviews = await answer_question(query, embedder, index, df, thinking_logger=thinking_logger, context=context)
            score = quality_score(query, response, reviews, thinking_logger)
            elapsed_time = thinking_logger.get_elapsed_time()
            return response, reviews, score, thinking_logger.get_thoughts(), elapsed_time
            
    except Exception as e:
        thinking_logger.log(f"Error: {str(e)}")
        print(traceback.format_exc())
        elapsed_time = thinking_logger.get_elapsed_time()
        return f"I encountered an error while processing your request: {str(e)}", None, None, thinking_logger.get_thoughts(), elapsed_time

# Streamlit UI
st.title("Spotify Review Q&A Chatbot")

# Initialize state
if "history" not in st.session_state:
    st.session_state.history = []
    
if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    st.session_state.index, st.session_state.embeddings = load_or_compute_index(reviews, st.session_state.embedder)

# Add conversation management
if "conversation_reset" not in st.session_state:
    st.session_state.conversation_reset = False

def display_chat():
    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.write(message['content'])
            
            # Show quality score if available
            if message["role"] == "assistant" and "score" in message and message["score"] is not None:
                st.write(f"(Quality Score: {message['score']:.2f})")
            
            # Show thinking process if available
            if message["role"] == "assistant" and "thinking" in message and message["thinking"]:
                with st.expander(f"Thoughts for {message['elapsed_time']:.2f} seconds"):
                    for thought in message["thinking"]:
                        st.write(thought)
            
            # Show retrieved reviews if available
            if "reviews" in message and message["reviews"]:
                with st.expander("Retrieved Reviews"):
                    for review in message["reviews"]:
                        st.write(f"- {review}")

# Add reset conversation button
if st.button("Reset Conversation"):
    st.session_state.history = []
    st.session_state.conversation_reset = True
    st.rerun()

# Display chat history
display_chat()

# Query input
query = st.chat_input("Ask a question about Spotify reviews:")

if query:
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        thinking_placeholder = st.empty()
        
        # Add the query to history
        st.session_state.history.append({"role": "user", "content": query})
        
        # Create a progress placeholder
        progress_placeholder = st.empty()
        
        async def run():
            start_time = time.time()
            
            # Create a spinner but keep the thinking_placeholder for elapsed time updates
            with st.spinner("Thinking..."):
                # Set up a task to update elapsed time while processing
                async def update_thinking_time():
                    while True:
                        elapsed = time.time() - start_time
                        thinking_placeholder.write(f"Thinking for {elapsed:.1f} seconds...")
                        await asyncio.sleep(0.1)
                
                # Start the thinking time updates as a background task
                thinking_task = asyncio.create_task(update_thinking_time())
                
                try:
                    # Process the query
                    response, reviews, score, thinking, elapsed_time = await process_query(
                        query, 
                        st.session_state.embedder, 
                        st.session_state.index, 
                        df
                    )
                finally:
                    # Cancel the thinking time update task when processing is done
                    thinking_task.cancel()
                
                # Split the response to show thought process in a different way
                response_parts = response.split("</think>")
                final_response = response_parts[-1] if len(response_parts) > 1 else response
                
                # Clear and update the message
                thinking_placeholder.empty()
                message_placeholder.write(final_response)
                
                # Show quality score if available
                if score is not None:
                    st.write(f"(Quality Score: {score:.2f})")
                
                # Show thinking process
                with st.expander(f"Thoughts for {elapsed_time:.2f} seconds"):
                    # If response was split with </think>, show the first part
                    if len(response_parts) > 1:
                        st.write(response_parts[0])
                    # Otherwise show the thinking log
                    else:
                        for thought in thinking:
                            st.write(thought)
                
                # Show Chatbot Workflow section
                with st.expander("Chatbot Workflow"):
                    for thought in thinking:
                        st.write(thought)
                
                # Show retrieved reviews if available
                if reviews:
                    with st.expander("Retrieved Reviews"):
                        for review in reviews:
                            st.write(f"- {review}")
                
                # Update history
                st.session_state.history.append({
                    "role": "assistant",
                    "content": final_response,
                    "reviews": reviews,
                    "score": score,
                    "thinking": thinking,
                    "elapsed_time": elapsed_time
                })
        
        # Run the async function
        asyncio.run(run())