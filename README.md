An Image to Product Embeddings search.

To run, first install the required libraries:
pip install -r requirements.txt

To build the embeddings, run the build_index.py file. For improved embedding creation, the extract_metadata.py file needs to be updated with as much detail as possible, contextually alinged with the Zarr catalog.
Embeddings will only be created once or needed when a new product is added.

To run the frontend on a local server, use uvicorn as below:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload