---
title: RAG QA System
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.11.0
app_file: app/app.py
pinned: false
---

# RAG QA — NLP Assignment 3

QA system built on 20 Wikipedia articles (AI/ML domain). Uses hybrid BM25 + semantic retrieval with RRF and cross-encoder reranking. Evaluated with LLM-as-Judge across 4 configs (chunking strategy × retrieval mode).