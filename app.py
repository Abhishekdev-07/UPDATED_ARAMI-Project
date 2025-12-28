from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests
import feedparser
# NEW: We use the dedicated news scraper library
from newspaper import Article 
import nltk

# Fix for newspaper library sometimes needing this
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_NAME = "facebook/bart-large-cnn"

print(f"Loading {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    print("✅ ARAMI Backend Ready.")
except Exception as e:
    print(f"❌ ERROR: Could not load model.\nDetails: {e}")

# --- 1. RECURSIVE SUMMARIZER ---
def recursive_summarize(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_len = inputs["input_ids"].shape[1]
    
    if input_len < 1000:
        return generate_summary(text)
    else:
        split_idx = len(text) // 2
        split_idx = text.rfind(' ', 0, split_idx)
        part1 = text[:split_idx]
        part2 = text[split_idx:]
        s1 = recursive_summarize(part1)
        s2 = recursive_summarize(part2)
        return generate_summary(s1 + " " + s2)

def generate_summary(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=200,      
            min_length=50,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- 2. IMPROVED SCRAPER (Using Newspaper3k) ---
# --- 2. API-BASED SCRAPER (Using Jina AI) ---
def scrape(url):
    print(f"Scraping via API: {url}")
    try:
        # We prepend "https://r.jina.ai/" to the URL.
        # This tells the Jina API to fetch and clean the page for us.
        api_url = f"https://r.jina.ai/{url}"
        
        # We don't need fake headers anymore. The API handles it.
        response = requests.get(api_url, timeout=10)
        
        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            return None
            
        # Jina returns the full article text in Markdown format.
        text = response.text
        
        # Sanity check: If the text is too short, it might be a login page.
        if len(text) < 200:
            return None
            
        # We limit it to ~4000 chars so we don't overwhelm the summarizer
        return text[:4000]

    except Exception as e:
        print(f"API Connection Error: {e}")
        return None

# --- 3. ENDPOINTS ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_feed', methods=['POST'])
def get_feed():
    topic = request.json.get('topic', 'technology')
    rss_url = f"https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en"
    
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
    
    try:
        response = requests.get(rss_url, headers=headers, timeout=5)
        feed = feedparser.parse(response.content)
    except Exception as e:
        return jsonify({"articles": []})
    
    articles = []
    for entry in feed.entries[:10]:
        image = None
        if 'media_content' in entry:
            image = entry.media_content[0]['url']
        
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "source": entry.source.title,
            "date": entry.published,
            "image": image
        })
    return jsonify({"articles": articles})

@app.route('/summarize', methods=['POST'])
def get_summary():
    data = request.json
    url = data.get('url')
    
    # Run the new scraper
    text = scrape(url)
    
    # HALLUCINATION GUARD
    # If scraper returns None, we tell the user honestly instead of letting AI guess.
    if not text: 
        return jsonify({"summary": "🔒 This website is blocking our AI. Try a different news source!"})
    
    summary = recursive_summarize(text)
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True, port=5000)