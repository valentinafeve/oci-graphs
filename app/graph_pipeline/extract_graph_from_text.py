from dotenv import load_dotenv
import os
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_oci import ChatOCIGenAI
import oracledb
from typing import Iterable, List, Tuple, Any
import re
import asyncio
from database.connection import get_conn

# Extract entites from text
async def extract_entities_from_text(text, graph_transformer):
    documents = [Document(page_content=text)]
    return await graph_transformer.aconvert_to_graph_documents(documents)

def _get(obj: Any, name: str, default=None):
        """Try attribute, then mapping key, else default."""
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict):
            return obj.get(name, default)
        return default

def _is_stringy(x: Any) -> bool:
    return isinstance(x, (str, bytes))

def parse_nodes(raw_nodes, NODE_RE) -> List[Tuple[str, str]]:
    """
    Accepts either:
    - str with repr-like Node(...) text
    - iterable of Node-like objects or dicts with fields id/type
    Returns: list[(vid, vlabel)] deduped by vid (first label wins)
    """
    out: List[Tuple[str, str]] = []
    seen = set()

    if _is_stringy(raw_nodes):
        for vid, vlabel in NODE_RE.findall(raw_nodes or ""):
            if vid not in seen:
                seen.add(vid)
                out.append((vid, vlabel))
        return out

    # Assume iterable of objects/dicts
    if isinstance(raw_nodes, Iterable):
        for n in raw_nodes:
            vid = _get(n, "id")
            vlabel = _get(n, "type")
            if vid is None or vlabel is None:
                # Try nested forms or alternate field names if your classes differ
                raise ValueError(f"Node missing id/type: {n!r}")
            if vid not in seen:
                seen.add(vid)
                out.append((vid, vlabel))
        return out

    raise TypeError("parse_nodes expected str or iterable of node-like objects")

def parse_edges(raw_edges, REL_RE) -> List[Tuple[str, str, str]]:
    """
    Accepts either:
    - str with repr-like Relationship(...) text
    - iterable of Relationship-like objects or dicts with fields:
        type, source(id/type), target(id/type)
    Returns: list[(elabel, src_vid, dst_vid)]
    """
    out: List[Tuple[str, str, str]] = []

    if _is_stringy(raw_edges):
        for src_vid, _src_type, dst_vid, _dst_type, elabel in REL_RE.findall(raw_edges or ""):
            out.append((elabel, src_vid, dst_vid))
        return out

    if isinstance(raw_edges, Iterable):
        for e in raw_edges:
            # Relationship may be object or dict
            elabel = _get(e, "type")
            src = _get(e, "source")
            dst = _get(e, "target")
            if elabel is None or src is None or dst is None:
                raise ValueError(f"Relationship missing type/source/target: {e!r}")
            src_vid = _get(src, "id")
            dst_vid = _get(dst, "id")
            if src_vid is None or dst_vid is None:
                raise ValueError(f"Relationship source/target missing id: {e!r}")
            out.append((elabel, src_vid, dst_vid))
        return out

    raise TypeError("parse_edges expected str or iterable of relationship-like objects")

def insert_into_oracle(nodes_out: List[Tuple[str, str]], edges_out: List[Tuple[str, str, str]], conn):
    with conn.cursor() as cur:
        # Upsert vertices (so re-runs don't fail)
        cur.executemany("""
            MERGE INTO vertices v
            USING (SELECT :1 AS vid, :2 AS vlabel FROM dual) s
            ON (v.vid = s.vid)
            WHEN MATCHED THEN UPDATE SET v.vlabel = s.vlabel
            WHEN NOT MATCHED THEN INSERT (vid, vlabel) VALUES (s.vid, s.vlabel)
        """, nodes_out)

        # Insert edges; if you want idempotency, you can change to MERGE on (elabel, src_vid, dst_vid)
        skipped = []
        try:
            cur.executemany(
                """
                INSERT INTO edges (elabel, src_vid, dst_vid)
                VALUES (:1, :2, :3)
                """,
                edges_out,
                batcherrors=True,            # keep going on row-level errors
                arraydmlrowcounts=True       # per-row success counts
            )

            # Report any failed rows (e.g., FK violations)
            errs = cur.getbatcherrors()
            if errs:
                for e in errs:
                    bad_row = edges_out[e.offset]     # the input row that failed
                    skipped.append((bad_row, e.message))
                    print(f"SKIPPED edge {bad_row} -> {e.message}")

            # Optional: how many actually inserted
            rowcounts = cur.getarraydmlrowcounts()
            inserted = sum(1 for c in rowcounts if c > 0)

            conn.commit()
            print(f"Inserted {inserted} edges; Skipped {len(skipped)} edges.")

        except oracledb.Error as ex:
            # Only fires for non row-level issues (e.g., SQL syntax, dead connection)
            conn.rollback()
            raise
        return skipped

def extract_graph_from_text(text: str, is_initial: bool):
    #####################
    # SETUP ENVIRONMENT #
    #####################

    # Load the .env file
    load_dotenv()

    # Set LLM with temperature=0 (less room for creativity)
    llm = ChatOCIGenAI(
        model_id=os.getenv("OCI_LLM_MODEL_OCID"),
        service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id=os.getenv("OCI_COMPARTMENT_OCID"),

    )

    # Instantiate LLMGraphTransformer to extract entities from the graph
    graph_transformer = LLMGraphTransformer(llm=llm)

    conn = get_conn()

    #####################
    # Extract Entities  #
    #####################

    graph_documents = asyncio.run(extract_entities_from_text(text, graph_transformer))

    ##############################
    # Create SQL Property Graph  #
    ##############################
    if is_initial:
        # Create vertex and edge tables using a JSON column to allow for schema flexibility when quering for properties
        ddl_vertices = """
        CREATE TABLE vertices (
        vid     VARCHAR2(200)                                       CONSTRAINT pk_vertices PRIMARY KEY,
        vlabel  VARCHAR2(50)    NOT NULL,
        props   JSON            DEFAULT NULL
        )
        """

        ddl_edges = """
        CREATE TABLE edges (
        eid     NUMBER          GENERATED BY DEFAULT AS IDENTITY    CONSTRAINT pk_edges PRIMARY KEY,
        elabel  VARCHAR2(50)    NOT NULL,
        src_vid VARCHAR2(200)   NOT NULL                            CONSTRAINT fk_edges_src REFERENCES vertices(vid),
        dst_vid VARCHAR2(200)   NOT NULL                            CONSTRAINT fk_edges_dst REFERENCES vertices(vid),
        props   JSON            DEFAULT NULL
        )
        """

        drop_edges    = "BEGIN EXECUTE IMMEDIATE 'DROP TABLE edges PURGE'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF; END;"
        drop_vertices = "BEGIN EXECUTE IMMEDIATE 'DROP TABLE vertices PURGE'; EXCEPTION WHEN OTHERS THEN IF SQLCODE != -942 THEN RAISE; END IF; END;"

        # Drop Vertex and Edge tables if they already exist and create them fresh
        with conn.cursor() as cur:
            # Drop if exist (idempotent)
            cur.execute(drop_edges)
            print("EDGES DROPPED")
            cur.execute(drop_vertices)
            print("NODES DROPPED")
            # Create fresh
            cur.execute(ddl_vertices)
            print("NODES CREATED")
            cur.execute(ddl_edges)
            print("EDGES CREATED")
            conn.commit()

        # Drop Graph if it already exists and create them fresh
        drop_pg = """
        BEGIN
        EXECUTE IMMEDIATE 'DROP PROPERTY GRAPH document_pg';
        EXCEPTION WHEN OTHERS THEN
        IF SQLCODE NOT IN (-2149, -942) THEN -- 2149: not found for graphs; 942: table not found (older)
            RAISE;
        END IF;
        END;
        """

        create_pg = """
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
        """

        with conn.cursor() as cur:
            try:
                cur.execute(drop_pg)
                print('GRAPH DROPPED')
            except Exception:
                pass
            cur.execute(create_pg)
            print('GRAPH CREATED')
            conn.commit()

    ##############################
    #  Insert Entities to Graph  #
    ##############################

    # --- keep the regex for when you *do* pass strings ---
    NODE_RE = re.compile(
        r"Node\(\s*id='(.*?)'\s*,\s*type='(.*?)'\s*,\s*properties=\{.*?\}\s*\)",
        flags=re.DOTALL,
    )
    REL_RE = re.compile(
        r"Relationship\(\s*"
        r"source=Node\(\s*id='(.*?)'\s*,\s*type='(.*?)'\s*,\s*properties=\{.*?\}\s*\)\s*,\s*"
        r"target=Node\(\s*id='(.*?)'\s*,\s*type='(.*?)'\s*,\s*properties=\{.*?\}\s*\)\s*,\s*"
        r"type='(.*?)'\s*,\s*properties=\{.*?\}\s*\)",
        flags=re.DOTALL,
    )

    parsed_nodes = parse_nodes(graph_documents[0].nodes, NODE_RE)
    parsed_edges = parse_edges(graph_documents[0].relationships, REL_RE)
    skipped_edges = insert_into_oracle(parsed_nodes, parsed_edges, conn)

    print('COMPLETE')
    print(f'skipped edges: {skipped_edges}')