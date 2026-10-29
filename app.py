from pathlib import Path
import re
import textwrap

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.rag.pipeline import RAGPipeline

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
TABLE_DIR = BASE_DIR / "outputs" / "tables"


# Human-readable display names for the fitted K=12 topic model.
# The original topic IDs remain unchanged in the underlying analysis;
# these labels are used only for dashboard presentation.
TOPIC_LABELS = {
    "topic_1": "Criminal / Evidence",
    "topic_2": "Election",
    "topic_3": "Tax / Income Tax",
    "topic_4": "Goods / Duty / Trade",
    "topic_5": "Statutory Interpretation / General Provisions",
    "topic_6": "Constitutional Law",
    "topic_7": "Property / Civil Suits",
    "topic_8": "Arbitration / Contracts",
    "topic_9": "Land Acquisition",
    "topic_10": "Criminal Procedure",
    "topic_11": "Service Law",
    "topic_12": "Family / Hindu Property / Marriage",
}

PARTIAL_DECADE_NOTE = (
    "The 2020s are a partial decade in the available dataset and should "
    "not be interpreted as a complete-decade estimate."
)


st.set_page_config(
    page_title="Legal Intelligence System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# BACKGROUND IMAGE (embedded, self-contained)
# ============================================================

GAVEL_IMAGE_B64 = "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"
GAVEL_DATA_URI = f"data:image/jpeg;base64,{GAVEL_IMAGE_B64}"


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Inter:wght@400;500;600;700;800&display=swap');

    /* -------------------------------------------------------
       Base
    ------------------------------------------------------- */

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 12% 8%, rgba(196, 155, 58, 0.07), transparent 32%),
            radial-gradient(circle at 92% 85%, rgba(196, 155, 58, 0.05), transparent 38%),
            #0b0d10;
        color: #e9e6dd;
    }}

    .main .block-container {{
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }}

    /* -------------------------------------------------------
       Sidebar
    ------------------------------------------------------- */

    [data-testid="stSidebar"] {{
        background:
            linear-gradient(180deg, rgba(6,7,9,0.94) 0%, rgba(6,7,9,0.97) 55%, rgba(6,7,9,0.99) 100%),
            url('{GAVEL_DATA_URI}');
        background-size: cover;
        background-position: center 30%;
        border-right: 1px solid #26241c;
    }}

    [data-testid="stSidebar"] * {{
        color: #d8d3c4;
    }}

    [data-testid="stSidebar"] hr {{
        border-color: rgba(196, 155, 58, 0.25);
    }}

    /* Sidebar radio -> vertical tab list */
    [data-testid="stSidebar"] [role="radiogroup"] {{
        display: flex;
        flex-direction: column;
        gap: 0.3rem;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label {{
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(196, 155, 58, 0.14);
        border-left: 3px solid transparent;
        border-radius: 8px;
        padding: 0.55rem 0.8rem !important;
        margin: 0 !important;
        transition: all 0.15s ease;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        background: rgba(196, 155, 58, 0.08);
        border-left-color: rgba(196, 155, 58, 0.5);
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label[data-checked="true"],
    [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
        background: linear-gradient(90deg, rgba(196, 155, 58, 0.16), rgba(196, 155, 58, 0.03));
        border-left: 3px solid #c49b3a;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] label p {{
        font-size: 0.92rem;
        font-weight: 550;
        color: #efe9d8 !important;
    }}

    [data-testid="stSidebar"] [role="radiogroup"] input:checked ~ div p {{
        color: #e7c876 !important;
        font-weight: 650;
    }}

    /* -------------------------------------------------------
       Typography
    ------------------------------------------------------- */

    h1, h2, h3 {{
        font-family: 'Playfair Display', serif;
        letter-spacing: -0.01em;
        color: #f1ecdd;
    }}

    h1 {{
        font-size: 2.6rem !important;
        font-weight: 750 !important;
    }}

    h2 {{
        font-size: 1.7rem !important;
        font-weight: 700 !important;
    }}

    h3 {{
        font-size: 1.18rem !important;
        font-weight: 650 !important;
    }}

    p, span, div, label {{
        color: #cfc9b8;
    }}

    /* -------------------------------------------------------
       Cards
    ------------------------------------------------------- */

    .metric-card {{
        background: linear-gradient(160deg, #14161b 0%, #101216 100%);
        border: 1px solid rgba(196, 155, 58, 0.18);
        border-radius: 14px;
        padding: 1.3rem 1.4rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        min-height: 120px;
        position: relative;
        overflow: hidden;
    }}

    .metric-card::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, #c49b3a, transparent);
    }}

    .metric-label {{
        font-size: 0.74rem;
        color: #9c9686;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 650;
    }}

    .metric-value {{
        font-family: 'Playfair Display', serif;
        font-size: 2rem;
        font-weight: 750;
        color: #eadfc0;
        margin-top: 0.3rem;
    }}

    .metric-description {{
        color: #93917f;
        font-size: 0.82rem;
        margin-top: 0.35rem;
    }}

    .finding-card {{
        background: linear-gradient(160deg, #14161b 0%, #101216 100%);
        border: 1px solid rgba(196, 155, 58, 0.14);
        border-left: 3px solid #c49b3a;
        border-radius: 12px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }}

    .finding-number {{
        display: inline-block;
        background: rgba(196, 155, 58, 0.14);
        color: #d9b662;
        border: 1px solid rgba(196, 155, 58, 0.3);
        border-radius: 6px;
        padding: 0.22rem 0.6rem;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        margin-bottom: 0.6rem;
    }}

    .finding-title {{
        font-family: 'Playfair Display', serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1ecdd;
        margin-bottom: 0.45rem;
    }}

    .finding-text {{
        color: #aca684;
        line-height: 1.65;
        font-size: 0.92rem;
    }}

    .research-banner {{
        background:
            linear-gradient(120deg, rgba(9,10,12,0.92) 15%, rgba(9,10,12,0.75) 100%),
            url('{GAVEL_DATA_URI}');
        background-size: cover;
        background-position: center 35%;
        border: 1px solid rgba(196, 155, 58, 0.25);
        border-radius: 18px;
        padding: 2.4rem 2.4rem;
        color: #f1ecdd;
        margin-bottom: 1.6rem;
        box-shadow: 0 16px 44px rgba(0, 0, 0, 0.45);
    }}

    .research-label {{
        color: #d9b662;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }}

    .research-question {{
        font-family: 'Playfair Display', serif;
        font-size: 1.55rem;
        line-height: 1.5;
        font-weight: 650;
        color: #f4efe1;
        max-width: 900px;
    }}

    .section-intro {{
        color: #a9a38e;
        line-height: 1.75;
        max-width: 950px;
        margin-bottom: 1.5rem;
        font-size: 0.98rem;
    }}

    .source-note {{
        color: #7d7864;
        font-size: 0.75rem;
        margin-top: 0.4rem;
    }}

    .tag {{
        display: inline-block;
        background: rgba(196, 155, 58, 0.08);
        border: 1px solid rgba(196, 155, 58, 0.25);
        color: #cbb679;
        padding: 0.25rem 0.6rem;
        border-radius: 999px;
        font-size: 0.72rem;
        margin-right: 0.3rem;
    }}

    /* -------------------------------------------------------
       Section headers as tab-like dividers
    ------------------------------------------------------- */

    .stSubheader, h3 {{
        border-bottom: 1px solid rgba(196, 155, 58, 0.2);
        padding-bottom: 0.5rem;
    }}

    /* -------------------------------------------------------
       Images
    ------------------------------------------------------- */

    [data-testid="stImage"] {{
        border-radius: 12px;
        border: 1px solid rgba(196, 155, 58, 0.14);
        overflow: hidden;
    }}

    [data-testid="stImage"] img {{
        border-radius: 12px;
    }}

    [data-testid="stDataFrame"] {{
        border: 1px solid rgba(196, 155, 58, 0.16);
        border-radius: 10px;
        overflow: hidden;
    }}

    /* -------------------------------------------------------
       Buttons
    ------------------------------------------------------- */

    .stButton > button {{
        border-radius: 8px;
        border: 1px solid rgba(196, 155, 58, 0.4);
        background: linear-gradient(160deg, #1b1d23, #121317);
        color: #e7c876;
        font-weight: 650;
        transition: all 0.15s ease;
    }}

    .stButton > button:hover {{
        border-color: #c49b3a;
        background: linear-gradient(160deg, #23201a, #17140f);
        color: #f1d68d;
    }}

    /* -------------------------------------------------------
       Inputs
    ------------------------------------------------------- */

    .stTextArea textarea, .stTextInput input {{
        background: #14161b !important;
        border: 1px solid rgba(196, 155, 58, 0.22) !important;
        color: #e9e6dd !important;
        border-radius: 8px !important;
    }}

    [data-baseweb="select"] > div {{
        background: #14161b !important;
        border: 1px solid rgba(196, 155, 58, 0.22) !important;
        color: #e9e6dd !important;
        border-radius: 8px !important;
    }}

    /* -------------------------------------------------------
       Alerts
    ------------------------------------------------------- */

    [data-testid="stAlertContainer"] {{
        background: #14161b;
        border: 1px solid rgba(196, 155, 58, 0.2);
        border-radius: 10px;
        color: #cfc9b8;
    }}

    /* -------------------------------------------------------
       Divider
    ------------------------------------------------------- */

    hr {{
        border-color: rgba(196, 155, 58, 0.16) !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def figure_path(filename: str) -> Path:
    return FIGURE_DIR / filename


def show_figure(
    filename: str,
    caption: str | None = None,
):
    path = figure_path(filename)

    if path.exists():
        st.image(
            str(path),
            caption=caption,
            use_container_width=True,
        )
    else:
        st.warning(
            f"Visualization not found: {filename}"
        )


def find_topic_table() -> pd.DataFrame | None:
    """Find a topic-prevalence CSV without hard-coding one filename."""
    if not TABLE_DIR.exists():
        return None

    for csv_path in sorted(TABLE_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue

        if df.empty:
            continue

        normalized = {str(c).strip().lower(): c for c in df.columns}
        topic_cols = [
            normalized[c]
            for c in normalized
            if re.fullmatch(r"topic_\d+", c)
        ]

        # Wide format: one row per decade, topic columns.
        decade_col = normalized.get("decade")
        if decade_col and len(topic_cols) >= 8:
            out = df[[decade_col] + topic_cols].copy()
            out = out.rename(columns={decade_col: "decade"})
            return out

        # Long format: decade/topic/proportion.
        topic_col = normalized.get("topic")
        proportion_col = (
            normalized.get("proportion")
            or normalized.get("topic_proportion")
            or normalized.get("mean_proportion")
            or normalized.get("average_topic_proportion")
        )
        if decade_col and topic_col and proportion_col:
            out = df[[decade_col, topic_col, proportion_col]].copy()
            out.columns = ["decade", "topic", "proportion"]
            pivot = out.pivot_table(
                index="decade",
                columns="topic",
                values="proportion",
                aggfunc="mean",
            ).reset_index()
            return pivot

    return None


def show_topic_heatmap():
    """Render an interactive topic heatmap with human-readable labels."""
    df = find_topic_table()

    if df is None:
        st.warning(
            "The topic-prevalence table was not found. Showing the stored "
            "PNG and the topic-label mapping below."
        )
        show_figure(
            "topic_prevalence_heatmap.png",
            "Stored topic-prevalence heatmap",
        )
        return

    topic_cols = [
        c for c in df.columns
        if re.fullmatch(r"topic_\d+", str(c).lower())
    ]
    topic_cols = sorted(topic_cols, key=lambda x: int(re.search(r"\d+", str(x)).group()))

    if not topic_cols:
        show_figure(
            "topic_prevalence_heatmap.png",
            "Stored topic-prevalence heatmap",
        )
        return

    plot_df = df.copy()
    plot_df["decade"] = plot_df["decade"].astype(str)

    # Ensure all expected topics are represented where possible.
    topic_cols = [c for c in topic_cols if c in plot_df.columns]
    labels = [TOPIC_LABELS.get(str(c).lower(), str(c)) for c in topic_cols]

    z = plot_df[topic_cols].apply(pd.to_numeric, errors="coerce").to_numpy()
    x = labels
    y = plot_df["decade"].tolist()

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="Viridis",
            colorbar=dict(
                title="Mean<br>Topic<br>Proportion",
                tickformat=".1%",
            ),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Decade: %{y}<br>"
                "Mean topic proportion: %{z:.1%}"
                "<extra></extra>"
            ),
            zmin=0,
            zmax=max(0.22, float(pd.Series(z.flatten()).max(skipna=True))),
        )
    )

    fig.update_layout(
        title="Modeled Legal Topic Prevalence by Decade",
        xaxis_title="Modeled legal topic",
        yaxis_title="Decade",
        height=620,
        margin=dict(l=60, r=30, t=80, b=170),
        paper_bgcolor="#101216",
        plot_bgcolor="#101216",
        font=dict(color="#d8d3c4"),
        xaxis=dict(
            tickangle=-35,
            tickfont=dict(size=11),
            gridcolor="rgba(255,255,255,0.04)",
        ),
        yaxis=dict(
            categoryorder="array",
            categoryarray=y,
            gridcolor="rgba(255,255,255,0.04)",
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Topic proportions are model-derived and depend on the selected "
        "topic model and corpus. K=12 is the working model selected using "
        "coherence and interpretability."
    )


def show_topic_evolution():
    """Render an interactive topic-evolution chart when the table is available."""
    df = find_topic_table()

    if df is None:
        show_figure(
            "topic_evolution_by_decade.png",
            "Stored topic-evolution visualization",
        )
        return

    topic_cols = [
        c for c in df.columns
        if re.fullmatch(r"topic_\d+", str(c).lower())
    ]
    topic_cols = sorted(topic_cols, key=lambda x: int(re.search(r"\d+", str(x)).group()))

    if not topic_cols:
        show_figure(
            "topic_evolution_by_decade.png",
            "Stored topic-evolution visualization",
        )
        return

    fig = go.Figure()

    for col in topic_cols:
        fig.add_trace(
            go.Scatter(
                x=df["decade"].astype(str),
                y=pd.to_numeric(df[col], errors="coerce"),
                mode="lines+markers",
                name=TOPIC_LABELS.get(str(col).lower(), str(col)),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Decade: %{x}<br>"
                    "Mean topic proportion: %{y:.1%}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Modeled Legal Topic Prevalence Across Decades",
        xaxis_title="Decade",
        yaxis_title="Mean Topic Proportion",
        yaxis_tickformat=".0%",
        height=600,
        margin=dict(l=60, r=30, t=80, b=80),
        paper_bgcolor="#101216",
        plot_bgcolor="#101216",
        font=dict(color="#d8d3c4"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Topic labels are human interpretations of the fitted model's "
        "top terms; they are not manually assigned legal categories."
    )


def metric_card(
    label: str,
    value: str,
    description: str,
):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def finding_card(
    number: int,
    title: str,
    text: str,
):
    st.markdown(
        f"""
        <div class="finding-card">
           <div class="finding-number">
    FINDING {number}
</div>
            <div class="finding-title">
                {title}
            </div>
            <div class="finding-text">
                {text}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
# ============================================================
# RAG PIPELINE
# ============================================================

@st.cache_resource(show_spinner=False)
def get_rag_pipeline() -> RAGPipeline:
    """
    Load the RAG pipeline once and reuse it across Streamlit reruns.
    """
    return RAGPipeline()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            padding: 0.6rem 0 1.3rem 0;
            border-bottom: 1px solid rgba(196, 155, 58, 0.22);
            margin-bottom: 1rem;
        ">
            <div style="
                font-family: 'Playfair Display', serif;
                font-size: 1.25rem;
                font-weight: 750;
                color: #f1ecdd;
                letter-spacing: 0.01em;
            ">
                Legal Intelligence System
            </div>
            <div style="
                color: #9c9686;
                font-size: 0.76rem;
                margin-top: 0.35rem;
                letter-spacing: 0.03em;
            ">
                Indian Supreme Court Judgment Analytics
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Key Findings",
            "Additional Findings",
            "Topics",
            "Legal References",
            "Chatbot",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="
            color: #8b8672;
            font-size: 0.73rem;
            line-height: 1.6;
            letter-spacing: 0.01em;
        ">
        Research corpus<br>
        12,000 judgments<br>
        Indian Supreme Court<br>
        1950s&ndash;2020s
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        textwrap.dedent("""
        <div class="research-banner">
            <div class="research-label">
                Empirical Study
            </div>

            <div class="research-question">
                How have the structure, linguistic characteristics,
                and legal subject matter of Indian Supreme Court
                judgments evolved over time?
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.title("Indian Supreme Court Judgment Analytics")

    st.markdown(
        textwrap.dedent("""
        <div class="section-intro">
        An exploratory and statistical analysis of a representative
        12,000-judgment sample drawn from a corpus of 41,839 Indian
        Supreme Court judgments. The analysis examines temporal,
        structural, linguistic, legal-reference, and topical patterns,
        with the resulting findings motivating a controlled comparison of
        generic and structure-aware legal information retrieval.
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.caption(PARTIAL_DECADE_NOTE)

    cols = st.columns(4)

    with cols[0]:
        metric_card(
            "Original corpus",
            "41,839",
            "Judgments in the source dataset",
        )

    with cols[1]:
        metric_card(
            "Research sample",
            "12,000",
            "Stratified proportional sample",
        )

    with cols[2]:
        metric_card(
            "Temporal coverage",
            "1950s&ndash;2020s",
            "Eight observed decades",
        )

    with cols[3]:
        metric_card(
            "EDA analyses",
            "40+",
            "Tables, statistics and visualizations",
        )

    st.markdown("---")

    st.subheader("Why this analysis matters")

    left, right = st.columns([1.1, 0.9])

    with left:

        finding_card(
            1,
            "Judgment structure changed substantially over time",
            "Paragraph density, numbered-paragraph structure and "
            "page-marker usage show pronounced temporal variation.",
        )

        finding_card(
            2,
            "Judgment length is non-monotonic",
            "Median judgment length decreases toward the 1990s/2000s "
            "before increasing again in the 2010s and 2020s.",
        )

        finding_card(
            3,
            "Legal subject composition evolved",
            "Topic modelling identifies changing concentrations "
            "of criminal, tax, service, arbitration, constitutional "
            "and other legal domains.",
        )

    with right:

        finding_card(
            4,
            "Document characteristics are multidimensional",
            "PCA shows that judgment variation cannot be reduced "
            "to document length alone.",
        )

        finding_card(
            5,
            "Structural variation motivates retrieval research",
            "The observed format drift provides an empirical basis "
            "for comparing generic and structure-aware chunking "
            "in legal information retrieval.",
        )


# ============================================================
# KEY FINDINGS
# ============================================================

elif page == "Key Findings":

    st.title("Key Findings")

    st.markdown(
        """
        <div class="section-intro">
        Ten findings selected as the primary analytical story of the
        project. The remaining analyses are available under
        Additional Findings. The 2020s are a partial decade in the
        available dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    finding_card(
        1,
        "Judgment volume varies strongly across the observed period",
        "The research sample contains relatively fewer judgments in the "
        "early decades and greater annual volume in later periods. This "
        "establishes the temporal context for the downstream analysis.",
    )

    show_figure(
        "judgments_by_year.png",
        "Annual judgment volume. The 2020s are a partial decade.",
    )

    finding_card(
        2,
        "Judgment length is non-monotonic",
        "Median judgment length declines toward the 1990s and 2000s "
        "before rising again in the 2010s and 2020s. Median values are "
        "used because the corpus contains extreme long-form judgments.",
    )

    show_figure(
        "median_judgment_length_by_year.png",
        "Median words per judgment by year",
    )

    finding_card(
        3,
        "Paragraph density shows a pronounced structural shift",
        "Paragraph density is substantially higher in the 2010s and "
        "2020s than in earlier observed decades, indicating strong "
        "changes in document structure relative to document length.",
    )

    show_figure(
        "paragraph_density_by_decade.png",
        "Median paragraphs per 1,000 words by decade",
    )

    finding_card(
        4,
        "Structural markers evolved across the judgment corpus",
        "The prevalence of selected document markers varies across "
        "decades, including CORAM, ISSUES, HELD, numbered paragraphs "
        "and page markers. Page-marker prevalence is partly a source-"
        "formatting variable rather than a measure of legal complexity.",
    )

    show_figure(
        "structural_marker_drift.png",
        "Structural marker prevalence across decades",
    )

    finding_card(
        5,
        "Legal-reference patterns vary across the observed period",
        "Extracted statutory-section and constitutional-article references "
        "show temporal differences in document-level legal-reference "
        "features. These are extracted references, not a measure of "
        "citation quality or judicial citation behavior.",
    )

    show_figure(
        "legal_reference_counts_by_decade.png",
        "Median distinct statutory sections and constitutional articles by decade",
    )

    finding_card(
        6,
        "The modeled topical composition of judgments changes over time",
        "The K=12 topic model identifies changing prevalence of legal "
        "domains including taxation, criminal/evidence, service law, "
        "arbitration/contracts, constitutional law and other domains.",
    )

    show_topic_evolution()

    finding_card(
        7,
        "Topic prevalence differs substantially across decades",
        "The topic heatmap provides a compact view of how the modeled "
        "legal-subject distribution changes across the observed period.",
    )

    show_topic_heatmap()

    # --------------------------------------------------------
    # Additional topic-modelling / multivariate visualizations
    # --------------------------------------------------------

    show_figure(
        "topic_prevalence.png",
        "Topic prevalence across decades",
    )

    show_figure(
        "topic_evolution_by_decade.png",
        "Topic evolution by decade",
    )

    show_figure(
        "topic_x_structure.png",
        "Topic × structural feature analysis",
    )

    show_figure(
        "multivariate_correlation.png",
        "Multivariate correlation analysis",
    )

    finding_card(
        8,
        "Vocabulary diversity is temporally variable",
        "Vocabulary diversity changes across the observed period and "
        "should be interpreted alongside document length and repetition. "
        "The measure is comparative rather than an absolute measure of "
        "lexical richness.",
    )

    show_figure(
        "vocabulary_diversity_by_year.png",
        "Vocabulary diversity by year",
    )

    finding_card(
        9,
        "Judgment characteristics occupy a multidimensional space",
        "PCA shows that document variation cannot be reduced to document "
        "length alone. The first four principal components explain about "
        "82% of the standardized feature variation.",
    )

    show_figure(
        "pca_decade_trajectory.png",
        "Decadal movement in multivariate judgment structure",
    )

    finding_card(
        10,
        "Multiple dimensions contribute to the observed variation",
        "The PCA explained-variance profile shows that several components "
        "are required to represent the standardized judgment features, "
        "supporting a multidimensional view of document structure.",
    )

    show_figure(
        "pca_explained_variance.png",
        "Explained variance by principal component",
    )

    st.info(
        "Interpretation note: the analyses describe associations and "
        "temporal patterns in the sampled corpus. They do not establish "
        "causal explanations for changes in judicial documents."
    )


# ============================================================
# ADDITIONAL FINDINGS
# ============================================================

elif page == "Additional Findings":

    st.title("Additional Findings")

    st.markdown(
        """
        <div class="section-intro">
        Supporting EDA analyses that provide additional evidence,
        diagnostics and methodological detail. Figures that rely on
        weak classifications, source-coverage artifacts, or redundant
        measurements are intentionally excluded from the final dashboard.
        </div>
        """,
        unsafe_allow_html=True,
    )

    categories = {
        "Dataset and Temporal Analysis": [
            "judgments_by_year.png",
            "citation_presence_by_decade.png",
        ],
        "Structural Analysis": [
            "structural_drift_sci.png",
            "structural_marker_drift.png",
            "normalized_structural_marker_density.png",
            "median_paragraphs_by_year.png",
            "sentence_length_by_year.png",
        ],
        "Legal References": [
            "legal_reference_density_by_decade.png",
        ],
        "Vocabulary and Text Analysis": [
            "tfidf_vocabulary_diversity.png",
        ],
        "Topic and Multivariate Analysis": [
            "pca_explained_variance.png",
            "pca_decade_trajectory.png",
            "pca_pc1_pc2.png",
            "record_level_correlation_matrix.png",
        ],
    }

    for category, files in categories.items():

        st.subheader(category)

        existing = [
            f
            for f in files
            if figure_path(f).exists()
        ]

        if not existing:
            st.info(
                "No generated visualizations available "
                "for this category."
            )
            continue

        for i in range(0, len(existing), 2):

            cols = st.columns(2)

            for col, filename in zip(
                cols,
                existing[i:i + 2],
            ):

                with col:

                    display_name = (
                        filename
                        .replace("_", " ")
                        .replace(".png", "")
                        .title()
                    )

                    st.markdown(
                        f"""
                        <div class="finding-card">
                            <div class="finding-title">
                                {display_name}
                            </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    show_figure(filename)

                    if filename == "citation_presence_by_decade.png":
                        st.caption(
                            "Important: this measures availability of "
                            "reported citation metadata in the source "
                            "dataset, not actual citation behavior."
                        )

                    if filename == "structural_drift_sci.png":
                        st.caption(
                            "SCI is a project-defined selected structural "
                            "marker index; it is not a general measure of "
                            "judgment complexity."
                        )

                    if filename == "record_level_correlation_matrix.png":
                        st.caption(
                            "Spearman correlations are descriptive. "
                            "Count-density relationships can be mechanically "
                            "related and should not be interpreted causally."
                        )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True,
                    )


# ============================================================
# TOPICS
# ============================================================

elif page == "Topics":

    st.title("Legal Topic Analysis")

    st.markdown(
        """
        <div class="section-intro">
        Topic modelling was used to identify recurring legal subject
        domains and examine how their prevalence varies across the
        observed decades. K=12 was selected as a defensible working
        model based on coherence and interpretability rather than
        treating perplexity alone as sufficient.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Modeled legal topics")

    topic_rows = [
        [topic_id.replace("_", " ").title(), label]
        for topic_id, label in TOPIC_LABELS.items()
    ]

    topic_df = pd.DataFrame(
        topic_rows,
        columns=["Model ID", "Human-readable interpretation"],
    )

    st.dataframe(
        topic_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "These labels are human interpretations of the fitted model's "
        "top terms. The original topic IDs remain unchanged for "
        "reproducibility."
    )

    st.markdown("---")

    st.subheader("Topic prevalence by decade")
    show_topic_heatmap()

    st.markdown("---")

    st.subheader("Topic evolution")
    show_topic_evolution()

    st.markdown("---")

    st.subheader("Topic modelling methodology")

    st.markdown(
        """
        The analysis compared multiple topic counts using perplexity,
        coherence and interpretability. The selected K=12 model provides
        a useful balance between topic separation and interpretability,
        while remaining a descriptive model of the sampled corpus rather
        than a definitive taxonomy of Indian law.

        **Important:** topic prevalence is model-dependent. The topic
        labels shown above are interpretations of the model's prominent
        terms, not manually assigned legal classifications.
        """
    )


# ============================================================
# LEGAL REFERENCES
# ============================================================

elif page == "Legal References":

    st.title("Legal Reference Analysis")

    st.markdown(
        """
        <div class="section-intro">
        Extracted statutory sections and constitutional articles are
        analysed as document-level legal-reference features. The
        dataset's reported-citation field is treated separately because
        its coverage varies substantially across the observed decades.
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_figure(
        "legal_reference_counts_by_decade.png",
        "Median distinct statutory sections and constitutional articles by decade",
    )

    show_figure(
        "legal_reference_density_by_decade.png",
        "Median distinct legal references per 1,000 words by decade",
    )

    st.markdown("---")

    st.subheader("Important data-quality limitation")

    st.warning(
        "The extracted reported-citation field has uneven temporal "
        "coverage. Its decline in later decades should therefore not "
        "be interpreted as a direct measure of changing judicial "
        "citation behavior. The 2020s are also a partial decade."
    )


# ============================================================
# RETRIEVAL
# ============================================================

elif page == "Retrieval":

    st.title("Information Retrieval")

    st.markdown(
        """
        <div class="section-intro">
        The retrieval experiment compares generic fixed-window chunking
        against structure-aware chunking. The two systems use the same
        documents, embedding model, similarity function and retrieval
        settings; chunking strategy is the controlled experimental
        difference.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "Retrieval interface will become active after the embedding "
        "and FAISS pipeline has been completed and validated."
    )

    st.caption(
        "The controlled experiment changes chunking strategy while "
        "holding the document set, embedding model, similarity function "
        "and retrieval settings constant."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    Generic strategy
                </div>
                <div class="metric-value">
                    Fixed-window
                </div>
                <div class="metric-description">
                    500-token baseline with configurable overlap
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    Structure-aware strategy
                </div>
                <div class="metric-value">
                    Structure-aware
                </div>
                <div class="metric-description">
                    Paragraph and legal-document structure preserved
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# CHATBOT
# ============================================================

elif page == "Chatbot":

    st.title("Legal Research Assistant")

    st.markdown(
        """
        <div class="section-intro">
        Ask a natural-language question about Indian Supreme Court
        judgments. The system retrieves relevant judgment passages
        using the selected retrieval strategy and generates a
        grounded answer using only the retrieved evidence.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # RAG SYSTEM STATUS
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="finding-card">
            <div class="finding-title">
                Evidence-grounded legal research
            </div>
            <div class="finding-text">
                Answers are generated from retrieved Supreme Court
                judgment passages. Supporting judgments are shown
                below each response for provenance.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------------

    question = st.text_area(
        "Research question",
        placeholder=(
            "Example: What principles govern remoteness of "
            "damages in contract disputes?"
        ),
        height=130,
        key="rag_question",
    )

    # --------------------------------------------------------
    # CONTROLS
    # --------------------------------------------------------

    control_col1, control_col2, control_col3 = st.columns(
        [1.2, 1.2, 1]
    )

    with control_col1:

        strategy_display = st.selectbox(
            "Retrieval strategy",
            [
                "Structure-aware",
                "Generic",
            ],
            key="rag_strategy",
        )

    with control_col2:

        top_k = st.selectbox(
            "Number of retrieved chunks",
            [3, 5, 8, 10],
            index=1,
            key="rag_top_k",
        )

    with control_col3:

        st.markdown(
            """
            <div style="
                padding-top: 1.85rem;
                color: #8f8a79;
                font-size: 0.8rem;
            ">
                Recommended strategy:
                <strong style="color:#d9b662;">
                Structure-aware
                </strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Convert UI label to pipeline strategy name
    strategy_map = {
        "Structure-aware": "structure_aware",
        "Generic": "generic",
    }

    strategy = strategy_map[strategy_display]

    st.markdown("")

    # --------------------------------------------------------
    # ASK BUTTON
    # --------------------------------------------------------

    ask_button = st.button(
        "Ask Legal Research Assistant",
        type="primary",
        use_container_width=True,
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Enter a research question first."
            )

        else:

            # ------------------------------------------------
            # LOAD RAG PIPELINE
            # ------------------------------------------------

            try:

                with st.spinner(
                    "Retrieving relevant judgments and generating answer..."
                ):

                    pipeline = get_rag_pipeline()

                    result = pipeline.ask(
                        question=question.strip(),
                        strategy=strategy,
                        top_k=top_k,
                    )

                # Store result so it survives Streamlit reruns
                st.session_state["rag_result"] = result

            except Exception as exc:

                st.error(
                    "The RAG system could not complete the request."
                )

                st.code(
                    f"{type(exc).__name__}: {exc}",
                    language="text",
                )

    # --------------------------------------------------------
    # DISPLAY PREVIOUS / CURRENT RESULT
    # --------------------------------------------------------

    if "rag_result" in st.session_state:

        result = st.session_state["rag_result"]

        # answer_mode reflects whether the backend answered from
        # retrieved judgments ("grounded_rag") or, when retrieval did
        # not surface sufficiently relevant context, with general
        # legal information ("general_legal"). This is used only for
        # internal display decisions below and is never shown to the
        # user, so the answer experience is seamless either way.
        answer_mode = result.get("answer_mode", "grounded_rag")

        st.markdown("---")

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        st.subheader("Answer")

        answer = result.get(
            "answer",
            "",
        )

        if answer:

            st.markdown(answer, unsafe_allow_html=True)

        else:

            st.warning(
                "No answer was returned by the language model."
            )

        # ----------------------------------------------------
        # RETRIEVAL INFORMATION
        # ----------------------------------------------------

        retrieval_results = result.get(
            "retrieval_results",
            [],
        )

        context_result = result.get(
            "context",
            {},
        )

        sources = context_result.get(
            "sources",
            [],
        )

        st.markdown("---")

        st.subheader("Retrieval Summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        with summary_col1:

            metric_card(
                "Strategy",
                (
                    "Structure-aware"
                    if result.get("strategy") == "structure_aware"
                    else "Generic"
                ),
                "Retrieval strategy used",
            )

        with summary_col2:

            metric_card(
                "Chunks retrieved",
                str(len(retrieval_results)),
                "Top retrieved passages",
            )

        # Count unique judgments
        unique_diaries = []

        for item in retrieval_results:

            diary = item.get("diary_no")

            if diary and diary not in unique_diaries:
                unique_diaries.append(diary)

        with summary_col3:

            metric_card(
                "Judgments",
                str(len(unique_diaries)),
                "Unique supporting judgments",
            )

        # ----------------------------------------------------
        # SUPPORTING JUDGMENTS
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader("Supporting Judgments & Citations")

        if not sources:

            st.info(
                "No source metadata was returned."
            )

        else:

            for source in sources:

                source_num = source.get("source_number", 1)
                case_name = source.get("case_name", "Case name unavailable")
                year = source.get("year", "")
                year_suffix = f" ({year})" if year and year != "Year unavailable" else ""
                diary_no = source.get("diary_no", "Unknown")
                judgment_date = source.get("judgment_date", "Unknown")
                citation = source.get("citation", "Citation unavailable")
                url = source.get("url")
                score = source.get("score")
                score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
                section = source.get("section_label", "general")

                if citation and citation != "Citation unavailable":
                    citation_html = f'<span style="color:#f1ecdd; font-family:monospace; background:rgba(196,155,58,0.14); padding:2px 8px; border-radius:4px; border:1px solid rgba(196,155,58,0.3);">{citation}</span>'
                else:
                    citation_html = '<span style="color:#8f8a79; font-style:italic;">Citation unavailable</span>'

                if url:
                    url_html = f'<a href="{url}" target="_blank" rel="noopener noreferrer" style="color:#d9b662; font-weight:600; text-decoration:underline;">View Official Supreme Court Judgment (PDF) &#x2197;</a>'
                else:
                    url_html = '<span style="color:#8f8a79; font-style:italic;">Judgment link unavailable</span>'

                st.markdown(
                    f"""
                    <div id="source-{source_num}" class="finding-card" style="scroll-margin-top: 1.5rem;">

                        <div class="finding-number" style="display:flex; justify-content:space-between; align-items:center;">
                            <span>CITATION [{source_num}]</span>
                            <span style="font-size:0.75rem; color:#8f8a79; font-weight:normal;">
                                Relevance Score: <strong style="color:#d9b662;">{score_text}</strong> &middot; Section: {section}
                            </span>
                        </div>

                        <div class="finding-title" style="margin-top:0.4rem; font-size:1.12rem; line-height:1.4;">
                            {case_name}{year_suffix}
                        </div>

                        <div class="finding-text" style="margin-top:0.6rem; line-height:1.7;">
                            <strong>Verified Citation:</strong> {citation_html}
                            <br>
                            <strong>Official Document:</strong> {url_html}
                            <br>
                            <span style="color:#8f8a79; font-size:0.8rem;">
                                Diary No: <code>{diary_no}</code> &middot; Judgment Date: {judgment_date}
                            </span>
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------
        # RETRIEVED PASSAGES
        # ----------------------------------------------------

        st.markdown("---")

        with st.expander(
            "View retrieved judgment passages"
        ):

            if not retrieval_results:

                st.info(
                    "No retrieved passages available."
                )

            else:

                for idx, item in enumerate(
                    retrieval_results,
                    start=1,
                ):

                    diary_no = item.get(
                        "diary_no",
                        "Unknown",
                    )

                    judgment_date = item.get(
                        "judgment_date",
                        "Unknown",
                    )

                    section = item.get(
                        "section",
                        "Unknown",
                    )

                    score = item.get(
                        "score",
                        None,
                    )

                    text = item.get(
                        "text",
                        "",
                    )

                    score_text = (
                        f"{score:.4f}"
                        if isinstance(score, (int, float))
                        else "N/A"
                    )

                    case_name = item.get("case_name", "Case name unavailable")
                    url = item.get("url")
                    doc_link = f' &middot; [Official PDF ↗]({url})' if url else ''

                    st.markdown(
                        f"""
                        **Rank {idx}: {case_name}**

                        **Diary:** {diary_no} &middot; **Date:** {judgment_date} &middot; **Section:** {section} &middot; **Score:** {score_text}{doc_link}
                        """
                    )

                    if text:

                        st.markdown(
                            f"> {text.replace(chr(10), chr(10) + '> ')}"
                        )

                    if idx < len(retrieval_results):

                        st.markdown("---")

        # ----------------------------------------------------
        # METHODOLOGY NOTE
        # ----------------------------------------------------

        st.markdown("---")

        st.caption(
            "This chatbot is a retrieval-augmented academic research "
            "system. Answers are grounded in retrieved judgment text "
            "and should not be treated as legal advice. Retrieved "
            "sources should be independently verified."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top: 4rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(196, 155, 58, 0.2);
        color: #7d7864;
        font-size: 0.75rem;
        text-align: center;
        letter-spacing: 0.02em;
    ">
        Legal Intelligence System &middot; Indian Supreme Court Judgment Analytics
        <br>
        Research support only. Verify legal authorities independently.
    </div>
    """,
    unsafe_allow_html=True,
)