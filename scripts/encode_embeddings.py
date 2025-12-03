import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def encode_sentences():
    """Encode sentences from answers.csv and save embeddings and index."""

    print("Loading data from data/data.csv...")
    df = pd.read_csv("../data/data.csv")

    # Ensure the output directory exists
    os.makedirs("../embeddings", exist_ok=True)

    #model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model_name = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  
    #model_name = "dangvantuan/sentence-camembert-large"

    print(f"Loading sentence transformer model: {model_name}...")
    model = SentenceTransformer(model_name)

    print("Encoding sentences...")
    embeddings = model.encode(df['sentence'].tolist(), show_progress_bar=True, convert_to_numpy=True)

    # Save embeddings
    print("Saving embeddings to ../embeddings/embeddings.npy...")
    np.save("../embeddings/embeddings.npy", embeddings.astype(np.float32))

    # Save index
    index_df = df[['participant', 'region', 'gender', 'sentence']].copy()
    index_df.reset_index(inplace=True)
    index_df.rename(columns={'index': 'row_index'}, inplace=True)
    index_df['doc_id'] = index_df['row_index']
    
    print("Saving index to ../embeddings/index.csv...")
    index_df.to_csv("../embeddings/index.csv", index=False)

    # Save model name
    os.makedirs("../configs", exist_ok=True)
    with open("../configs/embedding_model.txt", 'w') as f:
        f.write(model_name)

    print("Encoding complete.")

# Run the function
encode_sentences()