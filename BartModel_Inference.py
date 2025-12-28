from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests
import feedparser
from bs4 import BeautifulSoup
import time
import random

app = Flask(__name__)

# --- CONFIGURATION ---
MODEL_NAME = "facebook/bart-large-cnn"
print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()
print("✅ ARAMI Backend Ready.")

# --- 1. RECURSIVE SUMMARIZER (The "Smart Brain") ---
def recursive_summarize(text):
    # If text fits in memory, summarize it directly
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    input_len = inputs["input_ids"].shape[1]
    
    if input_len < 1000:
        return generate_summary(text)
    else:
        # Split into 2 halves
        split_idx = len(text) // 2
        split_idx = text.rfind(' ', 0, split_idx) # Find nearest space
        part1 = text[:split_idx]
        part2 = text[split_idx:]
        
        # Summarize parts and combine
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

# --- 2. SCRAPER ---
def scrape(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        response = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get Title & Text
        title = soup.find('h1').get_text().strip() if soup.find('h1') else ""
        paragraphs = [p.get_text().strip() for p in soup.find_all('p') if len(p.get_text()) > 50]
        full_text = f"Title: {title}. \n " + " ".join(paragraphs)
        
        return full_text
    except:
        return None

# --- 3. ENDPOINTS ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_feed', methods=['POST'])
def get_feed():
    topic = request.json.get('topic', 'technology')
    rss_url = f"https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    
    articles = []
    for entry in feed.entries[:10]: # Top 10 cards
        # Get image if available (some RSS feeds hide them, we'll try)
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
    text = scrape(url)
    
    if not text: 
        return jsonify({"summary": "I couldn't read this article (it might be blocked). Swipe to the next one!"})
    
    summary = recursive_summarize(text)
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True, port=5000)