import json
import os
import requests
import feedparser
from newspaper import Article
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time

# --- CONFIGURATION ---
DB_FILE = "news_db.json"
MODEL_NAME = "facebook/bart-large-cnn"

class NewsEngine:
    def __init__(self):
        print("🔧 Initializing News Engine...")
        self.load_model()
        self.db = self.load_db()

    def load_model(self):
        print(f"Loading AI Model ({MODEL_NAME})...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
            self.model.eval()
            print("✅ AI Model Loaded.")
        except Exception as e:
            print(f"❌ Model Error: {e}")

    def load_db(self):
        if os.path.exists(DB_FILE):
            with open(DB_FILE, 'r') as f:
                return json.load(f)
        return {"articles": []}

    def save_db(self):
        with open(DB_FILE, 'w') as f:
            json.dump(self.db, f, indent=4)
        print("💾 Database Saved.")

    # --- 1. THE ROBUST SCRAPER (Jina + Fallback) ---
    def scrape_article(self, url):
        print(f"   🔎 Scraping: {url[:50]}...")
        
        # STRATEGY A: Try Jina API (Fastest/Cleanest)
        try:
            jina_url = f"https://r.jina.ai/{url}"
            response = requests.get(jina_url, timeout=10)
            if response.status_code == 200 and len(response.text) > 500:
                return response.text[:4000] # Success!
        except:
            print("      ⚠️ Jina failed. Trying Backup...")

        # STRATEGY B: Newspaper3k (The Backup)
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
            # We fetch HTML manually to avoid blocking
            html = requests.get(url, headers=headers, timeout=10).content
            article = Article(url)
            article.download(input_html=html)
            article.parse()
            if len(article.text) > 200:
                return f"Title: {article.title}\n{article.text}"[:4000]
        except Exception as e:
            print(f"      ❌ Backup Scraper failed: {e}")
        
        return None

    # --- 2. SUMMARIZER ---
    def summarize_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=160,
                min_length=50,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # --- 3. THE MANAGER (Refreshes the DB) ---
    def refresh_feed(self, topic="Technology"):
        print(f"\n🔄 Refreshing News Feed for: {topic}")
        rss_url = f"https://news.google.com/rss/search?q={topic}&hl=en-US&gl=US&ceid=US:en"
        
        # Fetch RSS with User-Agent
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(rss_url, headers=headers)
        feed = feedparser.parse(response.content)

        new_articles = []
        
        # Process Top 5 Articles ONLY (To save time for now)
        for entry in feed.entries[:5]:
            # Check if we already have this article in DB (by link)
            existing = next((a for a in self.db['articles'] if a['link'] == entry.link), None)
            
            if existing and "summary" in existing:
                print(f"   ✅ Already in DB: {entry.title[:30]}...")
                new_articles.append(existing)
                continue

            # It's new! Let's process it.
            print(f"   ✨ Processing New: {entry.title[:30]}...")
            
            # Scrape & Summarize
            full_text = self.scrape_article(entry.link)
            
            if full_text:
                summary = self.summarize_text(full_text)
            else:
                summary = "Could not read article automatically. Tap 'Read More' to view."

            # Get Image
            image = entry.media_content[0]['url'] if 'media_content' in entry else None

            article_data = {
                "title": entry.title,
                "link": entry.link,
                "source": entry.source.title,
                "date": entry.published,
                "image": image,
                "summary": summary  # Saved FOREVER!
            }
            new_articles.append(article_data)

        # Update DB
        self.db['articles'] = new_articles
        self.save_db()
        return new_articles

# Helper for app.py to import
engine = NewsEngine()