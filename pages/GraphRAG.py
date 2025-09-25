# TO RUN: 
# conda activate graphrag
# python -m streamlit run app.py
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
import oracledb
import oci

# ---------- Config ----------
load_dotenv()
ORA_USER = os.getenv("ORA_USER")
ORA_PASS = os.getenv("ORA_PASS")
ORA_DSN  = os.getenv("ORA_DSN")
OCI_COMPARTMENT_OCID = os.getenv("OCI_COMPARTMENT_OCID")
OCI_LLM_MODEL_OCID   = os.getenv("OCI_LLM_MODEL_OCID")       # text-generation model

CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('~/.oci/config', CONFIG_PROFILE)
endpoint = "https://inference.generativeai.us-ashburn-1.oci.oraclecloud.com"
generative_ai_inference_client = oci.generative_ai_inference.GenerativeAiInferenceClient(config=config, service_endpoint=endpoint, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))
chat_detail = oci.generative_ai_inference.models.ChatDetails()

GRAPH_NAME = "document_pg"

# ---------- DB helpers ----------
@st.cache_resource(show_spinner=False)
def get_conn():
    # oracledb.init_oracle_client()  # if using Instant Client; else thin mode is fine without this
    os.environ["TNS_ADMIN"] = "/Users/rrtasker/Oracle/network/admin"  # your wallet dir
    return oracledb.connect(user=ORA_USER, password=ORA_PASS, dsn=ORA_DSN)
    

def run_query(sql: str, binds: dict | None = None) -> pd.DataFrame:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(sql, binds or {})
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchall() 
    return pd.DataFrame(rows, columns=cols)

def get_labels():
    conn = get_conn()
    with conn.cursor() as cur:
        node_label_query = """
            SELECT DISTINCT vid, vlabel FROM GRAPH_TABLE(document_pg
                MATCH (n)
                COLUMNS(n.vid as vid, n.vlabel as vlabel)
            )"""
        edge_label_query = """
            SELECT DISTINCT elabel FROM GRAPH_TABLE(document_pg
                MATCH () -[e]- ()
                COLUMNS(e.elabel as elabel)
            )
        """
        node_labels = cur.execute(node_label_query).fetchall()
        edge_labels = cur.execute(edge_label_query).fetchall()
        return [node_labels, edge_labels]

# ---------- GenAI helpers (skeletons you should implement with OCI SDK) ----------
def gen_plan_sql(user_question: str, labels: list) -> str:
    """
    Calls Oracle GenAI with a query-planner prompt and returns SQL/PGQ.
    For now, return a safe template if model not wired yet.
    """
    # print(labels)
    # OCI Generative AI call (Chat/Completions)
    # Use system prompt to guid the LLM
    # Provide labels/rels in user message.
    system_prompt = """
        You are a query planner for Oracle Database 23ai using SQL:2023 property graph patterns.
        Output ONE SQL statement that uses GRAPH_TABLE({GRAPH_NAME} MATCH ... COLUMNS(...)).

        Schema (Property Graph):
        CREATE PROPERTY GRAPH document_pg
            VERTEX TABLES ( vertices
                KEY (vid)
                LABEL nodes
                PROPERTIES (vid, vlabel, props)
            )
            EDGE TABLES ( edges
                KEY (eid)
                SOURCE KEY (src_vid) REFERENCES vertices(vid)
                DESTINATION KEY (dst_vid) REFERENCES vertices(vid)
                LABEL edges
                PROPERTIES (eid, src_vid, dst_vid, elabel, props)
            )
        Notes: 
        - All names of vertices are stored in vid
        - All node labels are stored in vlabel
        - All edge names are stored in elabel
        - props is a JSON column but is null, everything about a node is stored as a relationship
        - VID has """ + ",".join([item[0] for item in labels[0]]) + """
        - VLABEL has """ + ",".join([item[1] for item in labels[0]]) + """
        - ELABEL has """ + ",".join([item[0] for item in labels[1]]) + """

        Example Query:
        SELECT einstein_id, relationship, birthplace
        FROM GRAPH_TABLE (document_pg
            MATCH (n IS NODES WHERE n.vlabel = 'Person' AND n.vid = 'Albert Einstein' OR n.vid = 'Einstein') -[e IS EDGES WHERE e.elabel = 'BORN_IN']-> (m IS NODES)
            COLUMNS (n.vid AS einstein_id, e.elabel AS relationship, m.vid AS birthplace)
        )
        FETCH FIRST 100 ROWS ONLY;

        Rules:
        - Only SELECT (no DML/DDL).
        - Reference vertex table NODES and edge table EDGES.
        - Use column names exactly: vid, vlabel, eid, src_vid, dst_vid, elabel.
        - Prefer clear column aliases.
        - End with 'FETCH FIRST 100 ROWS ONLY'.
        - Do not include semicolons
        - Always use an alias for property names in the columns clause except for aggregations/functions
        - Multi-hop queries should follow a syntax like this example: -[t IS EDGES]->{1,4}
        Return ONLY SQL (no commentary).
        """.strip()
    
    # print(system_prompt)
    # seeds_txt = ", ".join(seed_entities) if seed_entities else "(none)"
    user_prompt = f"""User Question: {user_question}
        Available vertex labels: NODES
        Available edge types: EDGES
        """
    TextContent = oci.generative_ai_inference.models.TextContent
    Message = oci.generative_ai_inference.models.Message
    GenericChatRequest = oci.generative_ai_inference.models.GenericChatRequest
    OnDemandServingMode = oci.generative_ai_inference.models.OnDemandServingMode
    ChatDetails = oci.generative_ai_inference.models.ChatDetails

    system_msg = Message(
        role="SYSTEM",
        content=[TextContent(text=system_prompt)]
    )
    user_msg = Message(
        role="USER",
        content=[TextContent(text=user_prompt)]
    )

    chat_req = GenericChatRequest(
        api_format=GenericChatRequest.API_FORMAT_GENERIC,
        messages=[system_msg, user_msg],
        max_tokens=1800,
        temperature=0.2,
        top_p=0.9,
        top_k=20,
        presence_penalty=0.0,
        frequency_penalty=0.0,
    )

    chat_detail = ChatDetails(
        serving_mode=OnDemandServingMode(
            model_id=OCI_LLM_MODEL_OCID
        ),
        chat_request=chat_req,
        compartment_id=OCI_COMPARTMENT_OCID
    )

    resp = generative_ai_inference_client.chat(chat_detail)
    # Extract text
    try:
        sql = resp.data.chat_response.choices[0].message.content[0].text.strip()
    except Exception:
        raise RuntimeError(f"Bad response: {resp.data}")

    return sql

def gen_answer_from_results(question: str, rows: pd.DataFrame) -> str:
    """
    Calls Oracle GenAI to summarize/answer based strictly on rows+snippets.
    """
    # Provide the rows serialized as CSV/JSON
    if rows.empty:
        return "I didn’t find relevant connections for that query."
    else:
        system_prompt = """
            You are a data analyst answering a question for a business user. Given a dataset which is relavent to the question asked, answer the user's question.
            """.strip()

        user_prompt = f"""
            User Question: {question}
            Data Set: {rows}
            """
        TextContent = oci.generative_ai_inference.models.TextContent
        Message = oci.generative_ai_inference.models.Message
        GenericChatRequest = oci.generative_ai_inference.models.GenericChatRequest
        OnDemandServingMode = oci.generative_ai_inference.models.OnDemandServingMode
        ChatDetails = oci.generative_ai_inference.models.ChatDetails

        system_msg = Message(
            role="SYSTEM",
            content=[TextContent(text=system_prompt)]
        )
        user_msg = Message(
            role="USER",
            content=[TextContent(text=user_prompt)]
        )

        chat_req = GenericChatRequest(
            api_format=GenericChatRequest.API_FORMAT_GENERIC,
            messages=[system_msg, user_msg],
            max_tokens=1800,
            temperature=0.2,
            top_p=0.9,
            top_k=20,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        )

        chat_detail = ChatDetails(
            serving_mode=OnDemandServingMode(
                model_id=OCI_LLM_MODEL_OCID
            ),
            chat_request=chat_req,
            compartment_id=OCI_COMPARTMENT_OCID
        )

        resp = generative_ai_inference_client.chat(chat_detail)
        # Extract text
        try:
            answer = resp.data.chat_response.choices[0].message.content[0].text.strip()
        except Exception:
            raise RuntimeError(f"Bad response: {resp.data}")

        return answer


# ---------- Guardrails ----------
def is_sql_safe(sql: str) -> tuple[bool, str]:
    s = sql.strip().lower()
    # Allow only SELECT and MATCH, ban DDL/DML/PLSQL
    banned = ["insert", "update", "delete", "merge", "drop", "alter", "begin", "declare"]
    if not s.startswith("select"):
        return False, "Only SELECT statements are allowed."
    if any(b in s for b in banned):
        return False, "Statement contains disallowed keywords."
    if GRAPH_NAME.lower() not in s:
        return False, f"Query must reference the graph `{GRAPH_NAME}`."
    return True, ""

# ---------- Streamlit UI ----------
st.set_page_config(page_title="GraphRAG on Oracle 23ai", layout="wide")
st.sidebar.caption("Oracle Database 23ai • Property Graph • Oracle GenAI")

st.title("GraphRAG: Oracle Database 23ai + Oracle GenAI")

question = st.text_input("Ask a question about your knowledge graph:", placeholder="e.g., Who works with Larry Page at Company X?")

if st.button("Run", disabled=not question):
    with st.spinner("Thinking..."):
        sql = gen_plan_sql(question, get_labels())

        st.subheader("Generated graph query")
        st.code(sql, language="sql")
        ok, msg = is_sql_safe(sql)
        if not ok:
            st.error(f"Query rejected: {msg}")
        else:
            try:
                df = run_query(sql)
                st.subheader("Raw results")
                if df.empty:
                    st.info("No rows returned.")
                else:
                    st.dataframe(df, use_container_width=True)

                answer = gen_answer_from_results(question, df)
                st.subheader("LLM-generated answer")
                st.write(answer)

            except Exception as e:
                st.error(f"Execution error: {e}")
else:
    st.caption("Enter a question and click Run. We’ll show the LLM’s SQL/PGQ, execute it, and summarize the results.")

st.divider()
with st.expander("How it works"):
    st.markdown("""
1. The LLM generates a **SQL:2023 graph pattern** query (`MATCH ... ON GRAPH_NAME`).
2. We display the query, execute it, and show the rows.
3. The LLM writes a final answer grounded on those rows (and any snippets).
""")
