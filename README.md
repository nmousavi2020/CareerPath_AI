# CareerPath AI 🔀

**Academic vs. Industry Career Intelligence**

CareerPath AI is an ML-powered career intelligence application that analyzes professional CVs and academic profiles to evaluate alignment with academic and industry career paths.

### Machine Learning Techniques
- **Self-Supervised Graph Convolutional Network (GCN):** Learns node representations through graph link reconstruction.
- **Graph Embeddings:** Captures relationships between researchers, skills, research topics, and career paths.
- **Cosine Similarity:** Measures alignment between professional profiles and career directions.
- **Spectral Embedding:** Provides a fallback graph representation method when PyTorch is unavailable.

### Key Features
- CV-based professional skill extraction.
- Academic researcher data integration using the OpenAlex API.
- Academic vs. industry career alignment scores.
- Interactive knowledge graph visualization.
- Personalized career development recommendations.
- Downloadable career analysis report.
- Support for academic, industry, and combined professional profiles.

### Technologies
Python · Streamlit · PyTorch · Scikit-learn · NetworkX · Plotly · NumPy · Pandas · OpenAlex API

### Installation

```bash
git clone https://github.com/nmousavi2020/CareerPath_AI.git
cd CareerPath_AI
pip install -r requirements.txt
streamlit run app.py
```

### Machine Learning Note

The GCN is trained in a self-supervised manner using link reconstruction on the constructed profile graph. It learns graph representations and does not directly predict career success or future outcomes.

### License

Copyright (c) 2026 Naeim Mousavi. All rights reserved.

See `LICENSE` for licensing terms.