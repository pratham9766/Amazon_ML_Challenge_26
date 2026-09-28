"""Rare-token channel wrapper."""
from .tfidf_retrieval import retrieve_channel

def retrieve_tokens(db,queries,out,config,field):
    return retrieve_channel(db,queries,out,config,f"{field}_word")
