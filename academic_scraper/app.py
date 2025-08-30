import json
import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template_string
from celery import Celery
from celery.result import AsyncResult
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import nltk

# Initialize Flask and Celery
app = Flask(__name__)
celery = Celery(app.name)
app.config.from_mapping(
    CELERY=dict(
        broker_url="redis://localhost:6379/0",
        result_backend="redis://localhost:6379/0",
        task_ignore_result=False,
    ),
)
celery.conf.update(app.config["CELERY"])

# You may need to download NLTK data the first time you run this
# try:
#     nltk.data.find('tokenizers/punkt')
# except nltk.downloader.DownloadError:
#     nltk.download("punkt")
# try:
#     nltk.data.find('corpora/stopwords')
# except nltk.downloader.DownloadError:
#     nltk.download("stopwords")

# The HTML template as a string for simplicity
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Academic Paper Scraper & Summarizer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-slate-900 text-white min-h-screen flex items-center justify-center p-4">

    <div class="w-full max-w-2xl bg-slate-800 p-8 rounded-xl shadow-lg">
        <h1 class="text-3xl font-bold text-center mb-6">Academic Paper Summarizer</h1>
        <p class="text-slate-400 text-center mb-8">
            Enter a topic below to scrape academic papers and get a brief summary of each.
            This task runs in the background.
        </p>

        <form id="search-form" class="space-y-4">
            <input type="text" id="topic-input" name="topic" placeholder="e.g., 'Quantum Computing' or 'Machine Learning'"
                   class="w-full px-4 py-3 rounded-md bg-slate-700 border border-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
            <button type="submit"
                    class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-md transition duration-300 ease-in-out transform hover:scale-105 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-slate-800">
                Start Scraping
            </button>
        </form>
        
        <div id="status-container" class="mt-8 hidden">
            <div class="bg-blue-900 p-4 rounded-md text-sm text-blue-200">
                <p>Task ID: <span id="task-id"></span></p>
                <p id="task-status" class="mt-2 font-medium">Status: Pending...</p>
            </div>
        </div>

        <div id="results-container" class="mt-8 space-y-6 hidden">
            <h2 class="text-2xl font-semibold">Summarized Papers</h2>
        </div>

    </div>

    <script>
        document.getElementById('search-form').addEventListener('submit', function(e) {
            e.preventDefault();

            const topic = document.getElementById('topic-input').value;
            const form = new FormData();
            form.append('topic', topic);

            // Hide old results and show status container
            document.getElementById('status-container').classList.remove('hidden');
            document.getElementById('results-container').classList.add('hidden');
            document.getElementById('task-status').textContent = "Status: Submitting task...";
            
            fetch('/submit', {
                method: 'POST',
                body: form
            })
            .then(response => response.json())
            .then(data => {
                const taskId = data.task_id;
                document.getElementById('task-id').textContent = taskId;
                
                // Start polling for task status
                const pollInterval = setInterval(() => {
                    fetch(`/status/${taskId}`)
                    .then(res => res.json())
                    .then(statusData => {
                        document.getElementById('task-status').textContent = `Status: ${statusData.status}`;
                        if (statusData.status === 'ready') {
                            clearInterval(pollInterval);
                            displayResults(statusData.result.papers);
                            document.getElementById('results-container').classList.remove('hidden');
                        }
                    })
                    .catch(err => {
                        clearInterval(pollInterval);
                        document.getElementById('task-status').textContent = `Error: ${err.message}`;
                        console.error('Error polling status:', err);
                    });
                }, 3000); // Poll every 3 seconds
            })
            .catch(err => {
                document.getElementById('task-status').textContent = `Error: ${err.message}`;
                console.error('Error submitting form:', err);
            });
        });
        
        function displayResults(papers) {
            const container = document.getElementById('results-container');
            container.innerHTML = `<h2 class="text-2xl font-semibold mb-4">Summarized Papers</h2>`;

            if (papers.length === 0) {
                container.innerHTML += `<p class="text-slate-400">No papers found for this topic.</p>`;
                return;
            }

            papers.forEach(paper => {
                const paperDiv = document.createElement('div');
                paperDiv.classList.add('bg-slate-700', 'p-6', 'rounded-lg', 'shadow-md');
                paperDiv.innerHTML = `
                    <h3 class="text-xl font-bold text-blue-400 mb-2">${paper.title}</h3>
                    <p class="text-slate-300 mb-3">${paper.summary}</p>
                    <a href="${paper.link}" target="_blank" class="text-blue-400 hover:text-blue-300 transition duration-200 text-sm">Read full paper &rarr;</a>
                `;
                container.appendChild(paperDiv);
            });
        }
    </script>
</body>
</html>
"""

@celery.task(bind=True)
def scrape_and_summarize(self, topic):
    """
    Scrapes academic papers for a given topic and summarizes them.
    """
    self.update_state(state="PROGRESS", meta={"message": "Starting to scrape..."})
    search_url = f"https://arxiv.org/search/?query={topic.replace(' ', '+')}&searchtype=all"
    
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        papers = []
        for entry in soup.find_all("li", class_="arxiv-result"):
            title_element = entry.find("p", class_="title is-5 mathjax")
            link_element = entry.find("a", href=True, title="Abstract")
            
            if title_element and link_element:
                papers.append({
                    "title": title_element.get_text(strip=True),
                    "link": link_element["href"]
                })

        if not papers:
            return json.dumps({"status": "complete", "papers": []})
        
        self.update_state(state="PROGRESS", meta={"message": f"Found {len(papers)} papers. Starting summarization..."})
        
        summaries = []
        for paper in papers:
            try:
                parser = HtmlParser.from_url(paper["link"], Tokenizer("english"))
                stemmer = Stemmer("english")
                summarizer = LuhnSummarizer(stemmer)
                summarizer.stop_words = get_stop_words("english")
                
                summary_sentences = summarizer(parser.document, 3)
                summary_text = " ".join([str(s) for s in summary_sentences])
                
                summaries.append({
                    "title": paper["title"],
                    "link": paper["link"],
                    "summary": summary_text
                })
            except Exception as e:
                summaries.append({
                    "title": paper["title"],
                    "link": paper["link"],
                    "summary": f"Could not generate summary: {e}"
                })

        self.update_state(state="PROGRESS", meta={"message": "Summarization complete. Wrapping up..."})
        
        return json.dumps({"status": "complete", "papers": summaries})
        
    except requests.exceptions.RequestException as e:
        self.update_state(state="FAILURE", meta={"error": str(e)})
        return json.dumps({"status": "failure", "error": str(e)})

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/submit", methods=["POST"])
def submit_topic():
    topic = request.form.get("topic")
    if not topic:
        return jsonify({"error": "Topic is required"}), 400
    task = scrape_and_summarize.delay(topic)
    return jsonify({"task_id": task.id}), 202

@app.route("/status/<task_id>")
def get_task_status(task_id):
    task = AsyncResult(task_id, app=celery)
    if task.ready():
        result = task.result
        return jsonify({
            "status": "ready",
            "result": json.loads(result) if isinstance(result, str) else result
        })
    else:
        return jsonify({"status": "pending"}), 202

if __name__ == '__main__':
    app.run(debug=True)
