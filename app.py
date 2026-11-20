"""A beginner-friendly movie plot search app based on the course starter."""

import csv
from pathlib import Path

import chromadb
import streamlit as st
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# Keep the model and database ready when Streamlit reruns after a click.
@st.cache_resource
def prepare_search():
    # 1. Read real movie plots from the CSV file.
    data_file = Path(__file__).parent / "movies.csv"
    with data_file.open(encoding="utf-8", newline="") as file:
        movies = list(csv.DictReader(file))

    # 2. Split each plot into chunks. Keep the title and source with each chunk.
    chunks = []
    metadata = []
    for movie in movies:
        words = movie["plot"].split()
        for start in range(0, len(words), 80):
            chunks.append(" ".join(words[start : start + 80]))
            metadata.append({"title": movie["title"], "movie_id": movie["movie_id"]})

    # 3. Use the embedding model from the course starter notebook.
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    # Give vectors the same length so comparisons focus on their direction.
    embeddings = model.encode(chunks, normalize_embeddings=True)

    # 4. Store the chunks and their embeddings in Chroma.
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    collection = client.create_collection("movie_plots")
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadata,
        ids=[str(i) for i in range(len(chunks))],
    )
    return model, collection


def semantic_search(query):
    model, collection = prepare_search()

    # Embed the question with the same model and retrieve similar passages.
    query_embedding = model.encode([query], normalize_embeddings=True)
    return collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=3,
    )


def main():
    st.title("Movie Plot Semantic Search")
    st.write("Describe a story to find similar passages from 250 real movie plots.")
    st.caption("Results may contain spoilers. Similarity does not guarantee a good match.")

    with st.form("search"):
        query = st.text_input("What kind of story are you looking for?")
        search_clicked = st.form_submit_button("Search")

    if search_clicked:
        if query.strip():
            with st.spinner("Searching… The first search loads the model and indexes the plots."):
                results = semantic_search(query.strip())
            st.subheader("Top 3 matching passages")
            for i in range(len(results["documents"][0])):
                movie = results["metadatas"][0][i]
                st.subheader(movie["title"])
                st.write(results["documents"][0][i])
                st.markdown(f"[Wikipedia source](https://en.wikipedia.org/?curid={movie['movie_id']})")
                st.divider()
        else:
            st.info("Please enter a story description.")

    st.caption("Try: A detective investigating a murder • A family dealing with a secret")
    st.markdown(
        "Data: [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/), "
        "David Bamman, Brendan O'Connor, and Noah A. Smith; Wikipedia contributors. "
        "[CC BY-SA 3.0 US](https://creativecommons.org/licenses/by-sa/3.0/us/). "
        "250-movie subset; plots split into passages."
    )


if __name__ == "__main__":
    main()
