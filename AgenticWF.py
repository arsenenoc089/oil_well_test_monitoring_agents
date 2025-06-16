import re
import pandas as pd
import numpy as np
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import TypedDict, List, Tuple
from langgraph.graph import StateGraph
import openai
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt

#option 2
MODEL_NAME = "cross-encoder/nli-distilroberta-base"  # lighter version nli for cloud deployment
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #developed on macbook
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)

#load_dotenv()
#load from cloud
QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
collection_name = "well_test_embeddings_v5"
embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
#llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", openai_api_key=OPENAI_API_KEY, temperature=0.5)
llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0.5)

st.set_page_config(
    page_title="Well Watch",
    page_icon="🛢️",
    layout="wide"
)

class SupervisorState(TypedDict):
    prompt_text: str
    zonal_test_df: pd.DataFrame
    historical_interpretations: List[Tuple[str, float, float]]  # <-- Tuple of (text, ent_score, sim_score)
    final_interpretation: str

if 'log_df' not in st.session_state:
    st.session_state.log_df = pd.DataFrame(columns=[
        'Date', 'Well Name', 'Z1 BHP', 'Z2 BHP', 'Z3 BHP',
        'WT LIQ', 'WT Oil', 'WT WCT', 'Open Zone'
    ])

def generate_embedding(text: str) -> list[float]:
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def add_delta_rate_zscore(df, columns, window=3):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.sort_values(by=['Well Name', 'Date']).reset_index(drop=True)

    for col in columns:
        df[f"{col}_delta"] = df.groupby("Well Name")[col].diff().fillna(0)
        df[f"{col}_rate"] = df.groupby("Well Name")[col].pct_change().fillna(0) * 100

        rolling_mean = df.groupby("Well Name")[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        rolling_std = df.groupby("Well Name")[col].transform(lambda x: x.rolling(window=window, min_periods=1).std())
        df[f"{col}_zscore"] = ((df[col] - rolling_mean) / rolling_std).fillna(0)

    return df

def get_contradiction_score(premise: str, hypothesis: str) -> float:
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    return probs[0].item()  # contradiction score

def get_entailment_score(premise: str, hypothesis: str) -> float:
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
    return probs[2].item()

def generate_zone_test_summary(log_df, oil, log_change_thresh_oil,oil_jump_thresh_major, open_zones, oil_spike_flag):
    if log_df.empty:
        return "No data available for summary."

    summary_lines = []
    zone_avg_oils = {}

    for well_name, well_df in log_df.groupby('Well Name'):
        for zone in ['Z1', 'Z2', 'Z3']:
            zone_df = well_df[well_df['Open Zone'].str.upper().str.strip() == zone]
            if not zone_df.empty:
                avg_bhp = zone_df[f'{zone} BHP'].mean()
                avg_wc = zone_df['WT WCT'].mean()
                avg_oil = zone_df['WT Oil'].mean()
                avg_liq = zone_df['WT LIQ'].mean()
                zone_avg_oils[zone] = avg_oil  # Store for total calculation
                summary_lines.append(
                    f"For {zone} water cut is {avg_wc:.1f}, oil rate is {avg_oil:.1f}."
                )
        # Only add total oil if all zones are present
        if all(zone in zone_avg_oils for zone in ['Z1', 'Z2', 'Z3']):
            total_oil = sum(zone_avg_oils[zone] for zone in ['Z1', 'Z2', 'Z3'])/3
            oil_log_change = np.log(oil / total_oil)
            if len(open_zones)>1 :
                if oil_log_change>=oil_jump_thresh_major and oil_spike_flag: #oil rate still very high along with the high oil compared to previous test
                    summary_lines.append(
                        "Major increase in WT Oil (oil rate) compared to average oil rate from all zones combined."
                    )
                elif oil_log_change >= log_change_thresh_oil: #comingled flow
                    summary_lines.append(
                        "The WT Oil (oil rate) is significantly above the average oil rate from all zones combined."
                    )
            #summary_lines.append(f"Average WT Oil from all zones combined is {total_oil:.1f}. Average water cut from all zones combined is {total_wc:.1f}.")
    return "\n".join(summary_lines) if summary_lines else ""

def generate_zone_interpretation_text(row, prev_row=None, z_threshold=0.5, oil_jump_thresh_major=0.3, wc_jump_thresh=0.2,
                                      pct_dev_thresh=0.1, bhp_change_thresh_major = 0.2, log_df=None):
    """
    Generate open/shut interpretation for zones using % deviation and BHP Z-scores.
    Includes oil and water cut change from previous test.

    Parameters:
    - row: current test row (pd.Series)
    - prev_row: previous test row (pd.Series)
    - z_threshold: threshold Z-score above which zone is considered shut
    - oil_jump_thresh: threshold for oil rate % change
    - wc_jump_thresh: threshold for WC % change
    - pct_dev_thresh: percentage deviation threshold

    Returns:
    - str: combined interpretation text
    """

    bhp_dict = {zone: row[zone] for zone in ['Z1 BHP', 'Z2 BHP', 'Z3 BHP'] if pd.notnull(row[zone])}
    oil = row.get('WT Oil', np.nan)

    #description = [f"Well test dated {row['Date'].strftime('%Y-%m-%d')}:"] #date not helpful for context hence commented
    description = ["Description:"]
    interpretation = ["Interpretation:"]

    if len(bhp_dict) < 2:
        return "Insufficient BHP data to interpret zone behavior."

    bhp_values = list(bhp_dict.values())
    mean_bhp = np.mean(bhp_values)

    # Step 1: Check if all BHPs are within pct deviation from mean
    pct_devs = [abs(val - mean_bhp) / mean_bhp for val in bhp_values]
    open_zones = []

    if all(dev <= pct_dev_thresh for dev in pct_devs):
        interpretation.append(
            #f"→ All BHPs are within ±{pct_dev_thresh * 100:.1f}% of mean ({mean_bhp:.0f}) → assume all zones are **open**."
            f" All zones BHPs are close to the mean BHP hence assume all zones are open."
        )
        open_zones=['Z1', 'Z2', 'Z3'] #all zone open
    else:
        # Step 2: Use Z-score logic
        std_bhp = np.std(bhp_values)
        z_scores = {
            zone: (val - mean_bhp) / std_bhp
            for zone, val in bhp_dict.items()
        }
        for zone in bhp_dict:
            zone_id = re.search(r'(Z\d)', zone)
            zone_id = zone_id.group(1).upper() if zone_id else None
            z = z_scores[zone]
            status = "open" if z < -z_threshold else "shut-in"
            if status == "open":
                open_zones.append(zone)
            description.append(f"- {zone_id}: {status} (Z = {z:.2f})") #for better zonal status interpretation

    if len(open_zones) == 1:
        zone_id = re.search(r'(Z\d)', open_zones[0])
        zone_id = zone_id.group(1).upper() if zone_id else None
        #description.append(f"→ Only {zone_id} appears open; other zones are likely shut.")
        interpretation.append(f"Only one zone {zone_id} is open as evident from its lower BHP value compared to mean BHP value; all other zones are shut as evident from significant high BHP values compared to mean BHP value.")
        new_row = {
            'Date': row['Date'],
            'Well Name': row.get('Well Name', 'Unknown'),
            'Z1 BHP': row['Z1 BHP'],
            'Z2 BHP': row['Z2 BHP'],
            'Z3 BHP': row['Z3 BHP'],
            'WT LIQ': row.get('WT LIQ'),
            'WT Oil': row.get('WT Oil'),
            'WT WCT': row.get('WT WCT'),
            'Open Zone': zone_id
        }
        # Append new row to session state DataFrame
        st.session_state.log_df.loc[len(st.session_state.log_df)] = new_row
    else: #commented so that calculate change for all the well tests
        # Detect significant jump/drop in each zone's BHP
        if len(open_zones)>0 and len(open_zones)<3: #assuming 3 zone configuration
            zone_ids = [match for zone in open_zones for match in re.findall(r'Z\d+', zone)]
            interpretation.append(f"{zone_ids} zones are open as evident from their BHP values compared to mean BHP value.")
        log_change_thresh_bhp = 0.1  # Adjust based on empirical tuning
        for zone in ['Z1 BHP', 'Z2 BHP', 'Z3 BHP']:
            prev_val = prev_row[zone]
            curr_val = row[zone]
            if pd.notna(prev_val) and pd.notna(curr_val) and prev_val > 0 and curr_val > 0:
                log_change = np.log(curr_val / prev_val)
                if log_change>bhp_change_thresh_major and zone in open_zones: #major spikes in open zones
                    interpretation.append(
                        f"Major increase in BHP for open zone {zone} compared to the previous well test."
                    )
                elif abs(log_change) >= log_change_thresh_bhp:
                    direction = "increase" if log_change > 0 else "decrease"
                    interpretation.append(
                        f"Significant {direction} in {zone} compared to previous well test."
                    )
                    # description.append(
                    #     #f"→ Significant {direction} in {zone}: {prev_val:.0f} → {curr_val:.0f} "
                    #     f"{zone} (log change: {log_change:+.3f}) compared to previous well test." #commented as change is known from above line
                    # )
                # else:
                #     interpretation.append(
                #         f"Not a significant change in {zone} compared to previous well test."
                #     )

    # --- Oil rate log change --- only when more than 1 zone are open
    log_change_thresh_oil = 0.1
    oil_spike_flag=False
    if prev_row is not None:
        prev_oil = prev_row.get('WT Oil', np.nan)
        oil = row.get('WT Oil', np.nan)
        if pd.notnull(oil) and pd.notnull(prev_oil) and prev_oil > 0 and oil > 0:
            oil_log_change = np.log(oil / prev_oil)
            if oil_log_change>oil_jump_thresh_major: #major spikes are strong indicators
                interpretation.append(
                    "Major increase in WT Oil (oil rate) compared to the previous well test."
                )
                oil_spike_flag= True #pass info on the high oil rate to verify with the zonal avg
            elif abs(oil_log_change) >= log_change_thresh_oil:
                direction = "increase" if oil_log_change > 0 else "decrease"
                interpretation.append(
                    f"Significant {direction} in WT Oil compared to previous well test."
                )
            description.append(
                # f"→ Significant **{direction}** in Oil rate: {prev_oil:.0f} → {oil:.0f} "
                f"WT Oil (log change: {oil_log_change:+.3f}) compared to previous well test."
            )
            # else: #symantic search may confuse due to similar descriptions
            #     interpretation.append(
            #         "Not a significant change in Oil rate compared to previous well test."
            #     )

    # --- Water cut log change ---
    log_change_thresh_water_cut = 0.2
    if prev_row is not None:
        prev_wc = prev_row.get('WT WCT', np.nan)
        wc = row.get('WT WCT', np.nan)
        if pd.notnull(wc) and pd.notnull(prev_wc) and prev_wc > 0 and wc > 0:
            wc_log_change = np.log(wc / prev_wc)
            if abs(wc_log_change) >= log_change_thresh_water_cut:
                direction = "increase" if wc_log_change > 0 else "decrease"
                interpretation.append(
                    f"Significant {direction} in WT WCT (Water cut) compared to previous well test."
                )
                description.append(
                    # f"→ Significant **{direction}** in Water Cut: {prev_wc:.1f} → {wc:.1f} "
                    f"WT WCT (log change: {wc_log_change:+.3f}) compared to previous well test."
                )
            # else:
            #     interpretation.append(
            #         "Not a significant change in WT WCT (Water cut) compared to previous well test."
            #     )

    description.extend(interpretation) #add the interpretation to the end of the descriptions for better search
    if not st.session_state.log_df.empty:
        description.append(f"Following are the previous zonal test results, for you to understand the individual zone performance :")
        description.append(generate_zone_test_summary(st.session_state.log_df,oil, log_change_thresh_oil, oil_jump_thresh_major, open_zones, oil_spike_flag))
    else:
        description.append(f"No zonal test data is available.")

    return "\n".join(description)

def store_to_qdrant(anomaly_df: pd.DataFrame, explanations: list) -> None:
    """
    Store anomalies and their explanations into the Qdrant vector database.
    """
    collection_name = "anomalies"
    existing_collections = [c.name for c in qdrant_client.get_collections().collections]
    if collection_name not in existing_collections:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config={"size": 1536, "distance": "Cosine"}
        )
    points = []
    for idx, (i, row) in enumerate(anomaly_df.iterrows()):
        # well = row["Well Name"]
        # test_date = pd.to_datetime(row["Date"]).strftime("%Y%m%d")
        # wt_liq = f"{row['WT LIQ']:.2f}"  # ensures consistent format
        # Build custom unique ID
        # point_id = f"{well}_{test_date}_{wt_liq}"
        text = f"Anomaly at index {i}: {explanations[idx]}"
        vector = embedding_model.embed_query(text)

        points.append({
            "id": idx,
            "vector": vector,
            "payload": {
                "query": text,
                "explanation": explanations[idx],
                "data": row.to_dict()
            }
        })

    qdrant_client.upsert(collection_name=collection_name, points=points)

# ---- Agents ----
def describe_welltest(state: SupervisorState) -> SupervisorState:
    df=st.session_state["df_selected"]
    features = ['WT Oil', 'WT WCT', 'Z1 BHP', 'Z2 BHP', 'Z3 BHP']
    df_augmented = add_delta_rate_zscore(df, features)
    # Select the last row (the test to analyze)
    test_row = df_augmented.iloc[-1]
    prev_test_row = None
    if len(df_augmented) > 1:
        prev_test_row = df_augmented.iloc[-2]
    prompt_text = "" # default on first load
    if not prev_test_row.empty:
        prompt_text = generate_zone_interpretation_text(test_row, prev_test_row)
    with st.expander("Agent 1: Describe Selected Well Test", expanded=False):
        st.markdown("**Generated Prompt:**")
        st.code(prompt_text)
    return {**state, "prompt_text": prompt_text}

def kb_search(state: SupervisorState) -> SupervisorState:
    prompt_text = state["prompt_text"]
    embedding = generate_embedding(prompt_text)
    # Search top 10 to have enough candidates
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=embedding,
        limit=44,
    )
    interpretations_with_scores = []

    for res in search_results:
        interp_text = res.payload.get("interpretation", "")
        sim_score = res.score
        entailment_score = get_entailment_score(prompt_text, interp_text)
        interpretations_with_scores.append((interp_text, sim_score, entailment_score))

    interpretations_with_scores.sort(key=lambda x: x[2], reverse=True)
    # Show top 3
    historical_interps = []
    with st.expander("Agent 2: Find Matching Interpretations from Knowledge Base", expanded=False):
        st.markdown("**Historical Interpretations:**")
        for i, (text, sim_score, entailment_score) in enumerate(interpretations_with_scores[:3]):
            st.markdown(f"**{i + 1}. Entailment: {entailment_score:.3f}, Similarity: {sim_score:.3f}**")
            st.write(text)
            historical_interps.append((text, entailment_score, sim_score))

    if not historical_interps:
        historical_interps.append("No matches found from KB.")

    return {**state, "historical_interpretations": historical_interps}

def generate_interpretation(state: SupervisorState)->SupervisorState:
    prompt_text=state["prompt_text"]
    historical_interps=state["historical_interpretations"]
    llm_prompt = f"""You are an expert reservoir engineer. Your task is to analyze and interpret a new well test record.
    ---
    ### Following observations are calculated compared to previous welltest, by an anomaly function. Log change value indicates the quantitative deviations:
    {prompt_text}
    ---
    ### Historically similar interpretations from the knowledge base (based on entailment and similarity scores):
    """

    for i, (interp_text, entailment_score, sim_score) in enumerate(historical_interps):
        llm_prompt += f"\n\nInterpretation {i + 1} (entailment score: {entailment_score:.3f}) (similarity score: {sim_score:.3f}) :\n{interp_text}"

    llm_prompt += (
        "\n\n---\n"
        "### Your Interpretation:\n"
        "Based on the information provided above and your own analysis:\n\n"
        "- Provide **3 concise bullet points** covering:\n"
        # "  1. Status of each zone\n"
        # "  2. Phase of the well\n"
        # "  3. Any recommendations\n\n"
        "  1. Zone status (e.g. open, shut-in)\n"
        "  2. Phase of the well (e.g., transient, zonal test, commingled flow, zonal optimization, steady state post optimization, acid stimulation, steady state post acid stimulation)\n"
        "  3. Recommendations (e.g., shut-in, optimization, acid stimulation, continue to monitor)\n\n"
        "**Instructions:**\n"
        "- Do NOT mention units (e.g., psi, %, bbl, m³)\n"
        "- Keep total output under **150 words**\n"
        "- Focus on clear, actionable insights only."
    )
    with st.expander("Agent 3: Send Prompt to LLM ", expanded=False):
        st.markdown("**Prompt sent to LLM:**")
        st.code(llm_prompt)

    with st.spinner("Agent 3: Generating AI interpretation..."):
        response = llm.invoke(llm_prompt)
        llm_text = response.content if hasattr(response, 'content') else str(response)
        st.markdown("**AI Generated Interpretation of the Well Test:**")
        st.write(llm_text)
    return {**state, "final_interpretation": llm_text}

def visualize_stategraph(stategraph):
    G = nx.DiGraph()
    for node in stategraph.nodes:
        G.add_node(node)
    for src, dst in stategraph.edges:
        G.add_edge(src, dst)
    plt.figure(figsize=(6, 4), facecolor='black')
    pos = nx.spring_layout(G, seed=42, k=0.7)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, edge_color='white')
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=3000, node_shape='s', edgecolors='white')
    for node, (x, y) in pos.items():
        plt.text(
            x, y, str(node), fontsize=12, ha='center', va='center',
            bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.4'),
            color='white'
        )
    plt.title("StateGraph Visualization", color='white')
    plt.axis('off')
    plt.gca().set_facecolor('black')  # background for plot area
    st.pyplot(plt.gcf())

# ---- Build LangGraph ----
workflow = StateGraph(SupervisorState)
workflow.add_node("describe_welltest", describe_welltest)
workflow.add_node("kb_search", kb_search)
workflow.add_node("generate_interpretation", generate_interpretation)
workflow.set_entry_point("describe_welltest")
workflow.add_edge("describe_welltest", "kb_search")
workflow.add_edge("kb_search", "generate_interpretation")

#workflow.draw("well_test_graph")
app = workflow.compile()

# Dictionary mapping well names to CSV file paths
csv_files = {
    "cheetah-10": "data/cheetah-10.csv",
    #"cheetah-20": "data/cheetah-20.csv",
    "cheetah-90": "data/cheetah-90.csv"
}

# Dropdown for file selection
selected_file_key = st.selectbox("Select a well for blind test", list(csv_files.keys()))
selected_file_path = csv_files[selected_file_key]

if selected_file_path:
    df_full = pd.read_csv(selected_file_path)
else:
    st.stop()

with st.expander("Well Test Table", expanded=True):
    df_full['Date'] = pd.to_datetime(df_full['Date'], format='%d/%m/%Y', errors='coerce')
    df_full = df_full.sort_values('Date').reset_index(drop=True)
    st.dataframe(df_full)

selected_test = st.selectbox("Select a well test for analysis", df_full.index)
if selected_test in [0]:
    st.stop()
df = df_full.loc[:selected_test].copy()

st.session_state["df_selected"]=df

if st.session_state.log_df is not None and not st.session_state.log_df.empty:
    test_row = df.iloc[-1]
    selected_date = pd.to_datetime(test_row['Date'])
    st.session_state.log_df['Date'] = pd.to_datetime(st.session_state.log_df['Date'])
    latest_log_date = st.session_state.log_df['Date'].max()
    # If selected date is not after the latest date in log_df, trim it
    if selected_date <= latest_log_date:
        st.session_state.log_df = st.session_state.log_df[st.session_state.log_df['Date'] <= selected_date]

col1, col2 = st.columns([1, 1])
with col1:
    with st.expander("Well Test Trend", expanded=True):
        fig = go.Figure()
        # Primary lines
        fig.add_trace(go.Scatter(x=df['Date'], y=df['WT Oil'], mode='lines', name='WT Oil'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['WT LIQ'], mode='lines', name='WT LIQ'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['WT WCT'], mode='lines', name='WT WCT (%)'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Z1 BHP'], mode='lines', name='Z1 BHP'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Z2 BHP'], mode='lines', name='Z2 BHP'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Z3 BHP'], mode='lines', name='Z3 BHP'))
        # WT THP on secondary y-axis
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df['WT THP'],
            mode='lines', name='WT THP',
            yaxis='y2', line=dict(dash='dot', color='black')
        ))

        fig.update_layout(
            title="Well Test Parameters Over Time",
            xaxis_title="Date",
            yaxis=dict(title="WT Parameters"),
            yaxis2=dict(
                title="WT THP",
                overlaying='y',
                side='right',
                showgrid=False
            ),
            hovermode='x unified',
            template="plotly_white",
            height=450,
            showlegend=False  # <-- Hides the legend
        )

        st.plotly_chart(fig, use_container_width=True)

    st.session_state["df_selected"] = df

    if st.session_state.log_df is not None and not st.session_state.log_df.empty:
        test_row = df.iloc[-1]
        selected_date = pd.to_datetime(test_row['Date'])
        st.session_state.log_df['Date'] = pd.to_datetime(st.session_state.log_df['Date'])
        latest_log_date = st.session_state.log_df['Date'].max()
        # Trim log_df if needed
        if selected_date <= latest_log_date:
            st.session_state.log_df = st.session_state.log_df[
                st.session_state.log_df['Date'] <= selected_date
            ]

with col2:
    with st.expander("StateGraph Workflow", expanded=True):
        visualize_stategraph(workflow)

with st.spinner("Running agentic workflow of selected well test..."):
    initial_state = {
        "zonal_test_df": df,
        "prompt_text": "",
        "historical_interpretations": [],
        "final_interpretation": ""
    }

    for step in app.stream(initial_state):
        node = step.get("node")
        output = step.get("output", {})

