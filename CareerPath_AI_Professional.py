
import streamlit as st
import requests
import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import cosine_similarity

# PyTorch powers the self-supervised Graph Convolutional Network (GCN).
# If PyTorch is unavailable, the app transparently falls back to Spectral Embedding.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

st.set_page_config(
    page_title="CareerPath AI",
    page_icon="🔀",
    layout="wide"
)

# ---------------- CSS ----------------

st.markdown("""
<style>
:root {
    --cp-bg: #f4f7fb;
    --cp-surface: #ffffff;
    --cp-surface-2: #eef3f9;
    --cp-border: #d8e1ec;
    --cp-text: #172337;
    --cp-muted: #263449;
    --cp-accent: #465bd8;
    --cp-accent-2: #087f73;
}
html, body, [class*="css"] { font-size: 18px !important; }
.stApp { background: var(--cp-bg); color: var(--cp-text); }
.block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
[data-testid="stAppViewContainer"] > .main { background: var(--cp-bg); }
h1 { font-size: 2.35rem !important; font-weight: 800 !important; color: #14213a !important; }
h2 { font-size: 1.85rem !important; font-weight: 750 !important; color: #1b2b45 !important; }
h3 { font-size: 1.45rem !important; font-weight: 700 !important; color: #243752 !important; }
p, label, .stMarkdown, [data-testid="stWidgetLabel"] p,
[data-testid="stCaptionContainer"] p, .stCaption,
[data-testid="stMarkdownContainer"] li { color: var(--cp-text); font-size: 1rem !important; line-height: 1.65 !important; }
[data-testid="stCaptionContainer"] p, .stCaption { color: var(--cp-muted) !important; font-size: .92rem !important; }
.hero { background: linear-gradient(115deg, #e6edff 0%, #e8f5f3 100%); padding: 2.5rem 2.8rem; border-radius: 22px; border: 1px solid #cbd8ed; box-shadow: 0 10px 28px rgba(31, 55, 90, .08); }
.hero h1 { color: #172b50 !important; font-size: clamp(2.2rem, 4vw, 3.2rem) !important; margin-bottom: .4rem; }
.hero p { color: #344965 !important; font-size: 1.08rem !important; }
[data-testid="stVerticalBlockBorderWrapper"], .card { background: #fff; border-color: var(--cp-border) !important; border-radius: 16px; }
hr, [data-testid="stDivider"] { border-color: var(--cp-border) !important; }
.stTextInput input, .stTextArea textarea, [data-baseweb="input"] input, [data-baseweb="textarea"] textarea { background: #fff !important; color: #172337 !important; border: 1px solid #b9c8da !important; border-radius: 10px !important; font-size: 1rem !important; }
[data-testid="stSelectbox"] [data-baseweb="select"] > div { background: #fff; border-color: #b9c8da; color: #172337; border-radius: 10px; }
[data-testid="stWidgetLabel"] p { color: #243752 !important; font-weight: 700 !important; }
.stButton button, .stDownloadButton button { min-height: 50px; width: 100%; border: 1px solid #4258c7; border-radius: 11px; background: #465bd8; color: #fff !important; font-size: 1rem !important; font-weight: 750; }
.stButton button:hover, .stDownloadButton button:hover { background: #3549bd; border-color: #3549bd; }
.stButton button p, .stDownloadButton button p { color: #fff !important; font-size: 1rem !important; }
[data-testid="stMetric"] { background: #fff; border: 1px solid var(--cp-border); border-radius: 14px; padding: 1.1rem 1.2rem; box-shadow: 0 5px 16px rgba(31,55,90,.05); }
[data-testid="stMetricLabel"] p { color: #53647a !important; font-size: .95rem !important; }
[data-testid="stMetricValue"] { color: #172b50 !important; font-size: 2rem !important; font-weight: 800; }
[data-testid="stProgressBar"] > div { background: #e1e8f1; border-radius: 99px; }
[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #465bd8, #159b8b); border-radius: 99px; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stPlotlyChart"] { background: #fff; border: 1px solid var(--cp-border); border-radius: 14px; padding: .4rem; }
.footer { text-align: center; color: #53647a; padding: 2rem 1rem 1rem; margin-top: 1rem; border-top: 1px solid var(--cp-border); font-size: .95rem; line-height: 1.8; }
@media (max-width: 768px) { .block-container { padding: 1rem .8rem 2rem; } .hero { padding: 1.6rem 1.3rem; } html, body, [class*="css"] { font-size: 16px !important; } }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
<h1>🔀 CareerPath AI</h1>
<p>Academic vs Industry Career Intelligence</p>
<p>Discover your career direction using research data or your CV alone.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---------------- PROFILE INPUT ----------------

st.header("👤 Profile")

name = st.text_input(
    "Full Name",
    placeholder="Enter your full name"
)

scholar = st.text_input(
    "Google Scholar Profile URL (optional)",
    placeholder="Paste a Google Scholar link if you have one"
)

linkedin = st.text_input(
    "LinkedIn Profile URL (optional)",
    placeholder="Paste a LinkedIn link if you have one"
)

cv = st.text_area(
    "Professional Skills / CV Information",
    placeholder="Add your education, experience, research interests, technical skills and projects...",
    height=190
)

st.caption("Google Scholar and LinkedIn are optional. You can generate a report using your CV alone.")

professional_background = st.selectbox(
    "Professional Background",
    options=[
        "Academic Researcher",
        "Industry Professional",
        "Academic + Industry",
        "Not sure — analyze my CV"
    ],
    help=(
        "Choose the profile type that best describes you. "
        "Industry professionals are not required to have an OpenAlex profile."
    )
)

# ---------------- SKILL EXTRACTION ----------------

SKILLS = [
    "python", "machine learning", "deep learning",
    "data science", "sql", "java", "c++",
    "project management", "leadership",
    "communication", "research", "statistics",
    "tensorflow", "pytorch", "networkx",
    "graph neural networks", "cloud computing",
    "aws", "docker", "nlp", "computer vision",
    "data analysis", "business intelligence",
    "marketing", "finance", "excel"
]

def extract_skills(text):
    text = text.lower()
    return [
        skill for skill in SKILLS
        if skill in text
    ]

# ---------------- OPENALEX ----------------

def search_researcher(name):
    url = "https://api.openalex.org/authors"
    params = {
        "search": name,
        "per-page": 5
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )
    response.raise_for_status()

    return response.json().get("results", [])


# ---------------- GRAPH CONSTRUCTION ----------------

def build_graph(author, skills):

    G = nx.Graph()

    author_id = "Researcher"

    G.add_node(
        author_id,
        category="researcher"
    )

    topics = author.get("topics", [])

    topics = sorted(
        topics,
        key=lambda x: x.get("count", 0),
        reverse=True
    )[:8]

    G.add_node("Academic Career", category="career")
    G.add_node("Industry Career", category="career")

    G.add_edge(author_id, "Academic Career", weight=0.2)
    G.add_edge(author_id, "Industry Career", weight=0.2)

    for topic in topics:

        topic_name = topic.get("display_name")

        if not topic_name:
            continue

        node = "Topic: " + topic_name

        G.add_node(node, category="topic")

        G.add_edge(
            author_id,
            node,
            weight=1
        )

        G.add_edge(
            node,
            "Academic Career",
            weight=1
        )

    for skill in skills:

        node = "Skill: " + skill

        G.add_node(node, category="skill")

        G.add_edge(
            author_id,
            node,
            weight=1
        )

        G.add_edge(
            node,
            "Industry Career",
            weight=1
        )

    return G, topics


# ---------------- GRAPH EMBEDDINGS ----------------

def graph_embeddings(G):
    """Learn node representations with a self-supervised 2-layer GCN.

    The GCN is trained to reconstruct observed graph links (link reconstruction),
    not to predict career outcomes. SpectralEmbedding is a fallback if torch is absent.
    Returns (node_embeddings, node_order, model_description).
    """
    nodes = list(G.nodes())
    adjacency = nx.to_numpy_array(G, nodelist=nodes, weight="weight", dtype=np.float32)

    if len(nodes) < 3:
        return None, nodes, "Not enough graph nodes"

    if TORCH_AVAILABLE:
        try:
            torch.manual_seed(42)
            np.random.seed(42)
            n = len(nodes)

            # Five interpretable input features: four node categories + normalized degree.
            categories = ["researcher", "topic", "skill", "career"]
            features = np.zeros((n, len(categories) + 1), dtype=np.float32)
            degrees = np.sum(adjacency, axis=1)
            max_degree = float(degrees.max()) if degrees.size and degrees.max() > 0 else 1.0

            for i, node in enumerate(nodes):
                category = G.nodes[node].get("category", "topic")
                if category in categories:
                    features[i, categories.index(category)] = 1.0
                features[i, -1] = degrees[i] / max_degree

            # Add self-loops and symmetric degree normalization.
            adjacency_with_loops = adjacency + np.eye(n, dtype=np.float32)
            degree_matrix = np.sum(adjacency_with_loops, axis=1)
            inv_sqrt = np.power(np.maximum(degree_matrix, 1e-8), -0.5)
            normalized_adj = inv_sqrt[:, None] * adjacency_with_loops * inv_sqrt[None, :]

            x = torch.tensor(features, dtype=torch.float32)
            a_hat = torch.tensor(normalized_adj, dtype=torch.float32)
            target = torch.tensor((adjacency > 0).astype(np.float32), dtype=torch.float32)

            class GraphConvolution(nn.Module):
                def __init__(self, in_features, out_features):
                    super().__init__()
                    self.linear = nn.Linear(in_features, out_features)

                def forward(self, node_features, graph_matrix):
                    return graph_matrix @ self.linear(node_features)

            class GCNEncoder(nn.Module):
                def __init__(self, in_features, hidden=32, out_features=16):
                    super().__init__()
                    self.gc1 = GraphConvolution(in_features, hidden)
                    self.gc2 = GraphConvolution(hidden, out_features)
                    self.dropout = nn.Dropout(p=0.10)

                def forward(self, node_features, graph_matrix):
                    hidden = F.relu(self.gc1(node_features, graph_matrix))
                    hidden = self.dropout(hidden)
                    return self.gc2(hidden, graph_matrix)

            model = GCNEncoder(x.shape[1])
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
            positive_count = float(target.sum().item())
            negative_count = float(target.numel() - positive_count)
            pos_weight = torch.tensor(
                [min(negative_count / max(positive_count, 1.0), 10.0)],
                dtype=torch.float32
            )

            model.train()
            for _ in range(220):
                optimizer.zero_grad()
                z = model(x, a_hat)
                reconstructed_links = z @ z.T
                loss = F.binary_cross_entropy_with_logits(
                    reconstructed_links, target, pos_weight=pos_weight
                )
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                learned = model(x, a_hat).cpu().numpy()

            return learned, nodes, "Self-supervised 2-layer Graph Convolutional Network (GCN)"

        except Exception as exc:
            # Keep the app usable if a local PyTorch operation fails.
            st.warning(f"GCN could not run ({exc}). Falling back to Spectral Embedding.")

    try:
        model = SpectralEmbedding(
            n_components=min(2, len(nodes) - 1),
            affinity="precomputed",
            random_state=42
        )
        embeddings = model.fit_transform(adjacency)
        return embeddings, nodes, "Spectral Embedding fallback (PyTorch unavailable)"
    except Exception:
        return None, nodes, "Graph embeddings unavailable"


# ---------------- CAREER SCORING ----------------

def calculate_scores(author, skills, topics, embeddings, nodes):

    publications = author.get("works_count", 0)
    citations = author.get("cited_by_count", 0)

    h_index = author.get(
        "summary_stats", {}
    ).get("h_index", 0)

    academic_score = min(
        100,
        20
        + min(publications / 5, 25)
        + min(citations / 100, 25)
        + min(h_index * 2, 20)
        + min(len(topics) * 2, 10)
    )

    industry_score = min(
        100,
        15 + len(skills) * 10
    )

    # Graph representation similarity
    if embeddings is not None:

        index = {
            node: i
            for i, node in enumerate(nodes)
        }

        researcher = embeddings[index["Researcher"]]

        academic = embeddings[index["Academic Career"]]

        industry = embeddings[index["Industry Career"]]

        academic_sim = cosine_similarity(
            [researcher], [academic]
        )[0][0]

        industry_sim = cosine_similarity(
            [researcher], [industry]
        )[0][0]

        # Normalize similarity into 0-100
        academic_graph = (
            (academic_sim + 1) / 2
        ) * 100

        industry_graph = (
            (industry_sim + 1) / 2
        ) * 100

        academic_score = (
            0.7 * academic_score
            + 0.3 * academic_graph
        )

        industry_score = (
            0.7 * industry_score
            + 0.3 * industry_graph
        )

    return round(academic_score), round(industry_score)


# ---------------- GRAPH VISUALIZATION ----------------

def show_graph(G):

    pos = nx.spring_layout(
        G,
        seed=42,
        k=1.2
    )

    edge_x = []
    edge_y = []

    for u, v in G.edges():

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(
            width=1.5,
            color="#aab7c8"
        ),
        hoverinfo="none"
    )

    colors = {
        "researcher": "#facc15",
        "topic": "#38bdf8",
        "skill": "#a78bfa",
        "career": "#4ade80"
    }

    node_x = []
    node_y = []
    node_colors = []
    node_text = []

    for node in G.nodes():

        x, y = pos[node]

        node_x.append(x)
        node_y.append(y)

        category = G.nodes[node].get(
            "category", "topic"
        )

        node_colors.append(
            colors.get(category, "#ffffff")
        )

        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(
            size=22,
            color=node_colors,
            line=dict(
                width=1,
                color="#334155"
            )
        ),
        textfont=dict(
            color="#243752",
            size=14
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace]
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
        height=680,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ---------------- ANALYSIS ----------------

st.divider()

def build_manual_author(profile_name, cv_text, detected_skills):
    """Create a CV-based profile without assuming the person is an academic."""
    text = (cv_text or "").strip()
    topic_candidates = [
        "machine learning", "artificial intelligence", "data science",
        "deep learning", "computer vision", "natural language processing",
        "software engineering", "finance", "business intelligence",
        "statistics", "project management", "cloud computing", "research"
    ]
    topics = [
        {"display_name": topic.title(), "count": 1}
        for topic in topic_candidates
        if topic in text.lower()
    ]
    return {
        "id": "local-profile",
        "display_name": profile_name or "Professional",
        "works_count": 0,
        "cited_by_count": 0,
        "summary_stats": {"h_index": 0},
        "topics": topics,
        "_source": "manual"
    }


def show_openalex_candidate(candidate, index):
    """Show the source information used to help identify an OpenAlex author."""
    author_id = candidate.get("id", "")
    stats = candidate.get("summary_stats", {}) or {}
    institutions = candidate.get("last_known_institutions", []) or []
    institution_names = [
        item.get("display_name", "Unknown")
        for item in institutions
        if item.get("display_name")
    ]

    with st.expander(
        f"Candidate {index + 1}: {candidate.get('display_name', 'Unknown')}",
        expanded=(index == 0)
    ):
        c1, c2, c3 = st.columns(3)
        c1.metric("Publications", candidate.get("works_count", 0))
        c2.metric("Citations", candidate.get("cited_by_count", 0))
        c3.metric("H-Index", stats.get("h_index", 0))

        st.write("**Full name:**", candidate.get("display_name", "Not available"))
        st.write("**OpenAlex ID:**", author_id or "Not available")
        st.write(
            "**Known institution(s):**",
            ", ".join(institution_names) if institution_names else "Not available"
        )
        st.write("**ORCID:**", candidate.get("orcid") or "Not available")

        st.write("**Research topics:**")
        candidate_topics = candidate.get("topics", []) or []
        if candidate_topics:
            for topic in candidate_topics[:10]:
                st.write("-", topic.get("display_name", "Unknown"))
        else:
            st.write("Not available")

        if author_id:
            st.markdown(f"[Open original OpenAlex profile]({author_id})")


if st.button(" ANALYZE MY CAREER", use_container_width=True):
    profile_name = name.strip() or "Professional"
    detected_skills = extract_skills(cv)
    authors = []
    source_message = ""

    # Industry-only profiles do not need an academic database match.
    if professional_background == "Industry Professional":
        local_author = build_manual_author(profile_name, cv, detected_skills)
        st.session_state["authors"] = [local_author]
        st.session_state["profile_source"] = "manual"
        st.session_state["profile_mode"] = "Industry Professional"
        st.session_state.pop("confirmed_author", None)
        st.session_state.pop("result", None)

    else:
        # Academic, combined, and undecided profiles can search OpenAlex.
        if name.strip():
            try:
                authors = search_researcher(name.strip())
            except requests.RequestException:
                source_message = "OpenAlex could not be reached. You can continue using your CV."
            except Exception:
                source_message = "OpenAlex data is unavailable. You can continue using your CV."

        local_author = build_manual_author(profile_name, cv, detected_skills)

        if authors:
            st.session_state["authors"] = authors
            st.session_state["profile_source"] = "openalex"
        else:
            st.session_state["authors"] = [local_author]
            st.session_state["profile_source"] = "manual"
            source_message = source_message or (
                "No OpenAlex match was found. Your analysis can use your CV instead."
            )

        st.session_state["cv_author"] = local_author
        st.session_state["profile_mode"] = professional_background
        st.session_state.pop("confirmed_author", None)
        st.session_state.pop("result", None)

    st.session_state["skills"] = detected_skills
    st.session_state["profile"] = {
        "name": profile_name,
        "scholar": scholar.strip(),
        "linkedin": linkedin.strip(),
        "cv": cv,
        "professional_background": professional_background
    }

    if source_message:
        st.info(source_message)
    elif professional_background == "Industry Professional":
        st.success("Industry profile prepared using your CV and professional skills. OpenAlex is not required.")
    else:
        st.success("Search completed. Review the results and confirm the correct profile, or choose CV-only analysis.")


# ---------------- RESEARCHER / PROFESSIONAL VERIFICATION ----------------

if "authors" in st.session_state:
    authors = st.session_state["authors"]
    profile_source = st.session_state.get("profile_source", "manual")
    profile_mode = st.session_state.get("profile_mode", "Not sure — analyze my CV")

    st.subheader("🔎 Verify Your Professional Profile")

    if profile_source == "manual":
        author = authors[0]
        if profile_mode == "Industry Professional":
            st.info(
                "Industry profile selected. No academic publication profile is required. "
                "The report will use the CV and skills you provided."
            )
        else:
            st.info(
                "No OpenAlex profile is being used. The report will be based on your CV and detected skills."
            )

        st.write("**Profile name:**", author.get("display_name"))
        st.write("**Profile source:** CV / information entered by you")
        cv_confirmed = st.checkbox(
            "I confirm that the CV and professional information I entered are mine."
        )

        if st.button(
            "Generate Career Report",
            use_container_width=True,
            disabled=not cv_confirmed
        ):
            st.session_state["confirmed_author"] = author
            st.session_state["confirmed_profile_type"] = profile_mode
            skills = st.session_state["skills"]
            G, topics = build_graph(author, skills)
            embeddings, nodes, graph_model = graph_embeddings(G)
            academic, industry = calculate_scores(
                author, skills, topics, embeddings, nodes
            )
            st.session_state["result"] = {
                "author": author,
                "skills": skills,
                "topics": topics,
                "graph": G,
                "academic": academic,
                "industry": industry,
                "embeddings": embeddings,
                "nodes": nodes,
                "graph_model": graph_model,
                "profile_type": profile_mode
            }

    else:
        st.write(
            f"OpenAlex returned {len(authors)} candidate(s). "
            "A name match alone does not prove that a profile belongs to you."
        )

        for i, candidate in enumerate(authors):
            show_openalex_candidate(candidate, i)

        st.divider()
        st.subheader("Select the profile to use")

        options = {
            (
                f"{a.get('display_name', 'Unknown')} | "
                f"{a.get('works_count', 0)} publications | {a.get('id', 'No ID')}"
            ): a
            for a in authors
        }
        cv_author = st.session_state.get("cv_author")
        options["Use CV only — do not use an OpenAlex profile"] = cv_author

        selected_label = st.selectbox(
            "Choose your profile (or select CV-only analysis):",
            list(options.keys())
        )
        selected_author = options[selected_label]
        using_openalex = selected_author.get("_source") != "manual"

        if using_openalex:
            st.warning(
                "Before continuing, check the name, OpenAlex ID, ORCID, institution "
                "and research topics. Only confirm the profile if it is yours."
            )
            identity_confirmed = st.checkbox(
                "I checked the information and confirm this OpenAlex profile belongs to me."
            )
        else:
            st.info("The report will use your CV and skills, not the OpenAlex candidate data.")
            identity_confirmed = st.checkbox(
                "I confirm that the CV and professional information I entered are mine."
            )

        if st.button(
            "Generate Career Report",
            use_container_width=True,
            disabled=not identity_confirmed
        ):
            selected_profile_type = (
                "Academic + Industry" if profile_mode == "Academic + Industry"
                else "Academic Researcher" if using_openalex
                else profile_mode
            )
            st.session_state["confirmed_author"] = selected_author
            st.session_state["confirmed_profile_type"] = selected_profile_type
            skills = st.session_state["skills"]
            G, topics = build_graph(selected_author, skills)
            embeddings, nodes, graph_model = graph_embeddings(G)
            academic, industry = calculate_scores(
                selected_author, skills, topics, embeddings, nodes
            )
            st.session_state["result"] = {
                "author": selected_author,
                "skills": skills,
                "topics": topics,
                "graph": G,
                "academic": academic,
                "industry": industry,
                "embeddings": embeddings,
                "nodes": nodes,
                "graph_model": graph_model,
                "profile_type": selected_profile_type
            }


# ---------------- RESULTS ----------------

if "result" in st.session_state:

    result = st.session_state["result"]

    author = result["author"]
    skills = result["skills"]
    topics = result["topics"]
    G = result["graph"]

    st.divider()

    st.header("📊 Career Intelligence Report")

    st.caption(
        "Experimental career-alignment indicators, not probabilities of future success. "
        "The graph model and data sources are disclosed below."
    )

    profile_type = result.get("profile_type", st.session_state.get("confirmed_profile_type", "Not specified"))
    st.write("**Professional profile type:**", profile_type)

    if author.get("_source") == "manual":
        st.info(
            "Data source: CV and skills entered by you. No OpenAlex researcher profile "
            "was used; publication and citation metrics are not available for this analysis."
        )
    else:
        st.caption("Data source: OpenAlex public researcher metadata, combined with your CV skills.")
        st.write("**Selected OpenAlex profile:**", author.get("display_name", "Not available"))
        st.write("**OpenAlex ID:**", author.get("id", "Not available"))
        if author.get("id"):
            st.markdown(f"[View original OpenAlex profile]({author.get('id')})")

    c1, c2 = st.columns(2)

    with c1:

        st.metric(
            "🎓 Academic Alignment",
            f"{result['academic']}%"
        )

        st.progress(
            result["academic"] / 100
        )

    with c2:

        st.metric(
            "💼 Industry Alignment",
            f"{result['industry']}%"
        )

        st.progress(
            result["industry"] / 100
        )

    if result["academic"] > result["industry"]:
        st.success(
            "Your current profile shows stronger academic alignment."
        )
    elif result["industry"] > result["academic"]:
        st.success(
            "Your current profile shows stronger industry alignment."
        )
    else:
        st.info("Your career alignment is balanced.")

    st.divider()

    st.header("🕸️ Knowledge Graph")

    st.write(
        "Connections between your research topics, "
        "professional skills and career directions."
    )

    show_graph(G)

    st.caption(
        "Yellow: Candidate | Blue: Research Topics | "
        "Purple: Skills | Green: Career Paths"
    )

    st.info(
        "Graph model: " + result.get("graph_model", "Not recorded") +
        ". The GCN learns node representations by reconstructing links in this profile graph; "
        "it is not a trained model of future career success."
    )

    st.divider()

    st.header("📚 Research & Professional Profile")

    if author.get("_source") == "manual":
        st.info("This profile is CV-based. Academic publication metrics are not inferred or fabricated.")
        h_index = 0
    else:
        p1, p2, p3 = st.columns(3)
        p1.metric("Publications", author.get("works_count", 0))
        p2.metric("Citations", author.get("cited_by_count", 0))
        h_index = author.get("summary_stats", {}).get("h_index", 0)
        p3.metric("H-Index", h_index)

    st.write("**Research Topics**")

    for topic in topics:

        st.write(
            "- " + topic.get("display_name", "")
        )

    st.write("**Detected Skills**")

    if skills:
        for skill in skills:
            st.write("- " + skill)
    else:
        st.info(
            "No matching skills detected. Add more information to your CV."
        )

    st.divider()

    st.header("💡 Career Development Recommendations")

    recommendations = []

    if result["academic"] >= result["industry"]:

        recommendations.extend([
            "Continue publishing in relevant research areas.",
            "Build international research collaborations.",
            "Develop practical projects based on your research.",
            "Improve communication of research impact."
        ])

    else:

        recommendations.extend([
            "Build an industry-focused portfolio.",
            "Develop practical projects using your research skills.",
            "Improve professional networking.",
            "Gain experience with real-world business problems."
        ])

    if len(skills) < 4:

        recommendations.append(
            "Add more verified technical and professional skills to your profile."
        )

    for item in recommendations:
        st.write("✅ " + item)

    st.divider()

    st.header("📥 Download Your Report")

    report = f"""
CAREERPATH AI
Developed by Naeim Mousavi

Researcher / Professional: {author.get('display_name')}
Professional Profile Type: {result.get('profile_type', 'Not specified')}
Data source: {"CV-based profile" if author.get("_source") == "manual" else "OpenAlex + CV"}
OpenAlex ID: {author.get('id', 'Not applicable')}
OpenAlex Profile: {author.get('id', 'Not applicable') if author.get('_source') != 'manual' else 'Not applicable'}
Graph Model: {result.get('graph_model', 'Not recorded')}

Academic Alignment: {result['academic']}%
Industry Alignment: {result['industry']}%

Publications: {author.get('works_count', 0)}
Citations: {author.get('cited_by_count', 0)}
H-Index: {h_index}

Research Topics:
{chr(10).join(t.get('display_name', '') for t in topics)}

Skills:
{', '.join(skills)}

Recommendations:
{chr(10).join(recommendations)}

Note: These scores represent experimental career alignment
indicators, not probabilities of future success.
"""

    st.download_button(
        "⬇️ Download Career Report",
        data=report,
        file_name="CareerPath_AI_Report.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.caption(
        "Graph representation uses the model identified above. The GCN is trained "
        "self-supervised on the current profile graph through link reconstruction. "
        "It is experimental and does not establish causal relationships or predict career success."
    )

# ---------------- FOOTER ----------------

st.markdown("""
<div class="footer">
CareerPath AI<br>
Developed by Naeim Mousavi<br>
© 2026 All Rights Reserved
</div>
""", unsafe_allow_html=True)
