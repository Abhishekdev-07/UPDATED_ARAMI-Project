from flask import Flask, render_template, request, jsonify
# Import the function from our new data engine
from Database_News import search_chronicling_america

app = Flask(__name__)

@app.route('/')
def index():
    """Serves the main landing page."""
    return render_template('index.html')

@app.route('/search')
def search():
    """
    This endpoint receives a search query from the frontend,
    calls our data engine to get live, summarized results from the
    Chronicling America API, and returns them as JSON.
    """
    # Get the search query and page number from URL parameters
    query = request.args.get('q', default='history', type=str)
    page = request.args.get('page', default=1, type=int)
    
    try:
        # Call our engine function to get the processed articles
        articles = search_chronicling_america(query, page)
        return jsonify(articles)
    except Exception as e:
        # Return an error message if the backend engine fails
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Setting debug=False because the model is loaded once, and reloading on
    # code changes can cause issues. Set to True for UI changes only.
    app.run(debug=False, port=5000)