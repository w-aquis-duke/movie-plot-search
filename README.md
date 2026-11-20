# Movie Plot Semantic Search

Search 250 movie plots by describing a story. Built with Streamlit, Chroma, and the MiniLM embedding model from the course starter.

The app splits plots into 80-word chunks, embeds them, and returns the three closest passages. Results may include spoilers or irrelevant matches.

## Run locally

Use Python 3.12:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first search downloads the model and builds the database with no API key needed.

## Deploy

Upload the project files to GitHub. In [Streamlit Community Cloud](https://share.streamlit.io/), select the repository, set the main file to `app.py`, choose Python 3.12, and deploy.

## Data

Movie plots come from the CMU Movie Summary Corpus. See [DATA_SOURCE.md](DATA_SOURCE.md) for attribution and license details.
