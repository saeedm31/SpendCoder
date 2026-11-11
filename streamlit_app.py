"""SpendCoder control panel: run and watch the whole process from one place.

    streamlit run streamlit_app.py

Pages: overview, data, train (baseline here; fine-tuning in Colab, on AWS SageMaker or locally), evaluate,
classify, review queue (people correct the unsure lines), deploy (promote a model, restart the API).

Anyone can try the demo. Actions that cost money or change production (big downloads, SageMaker jobs,
promoting a model, restarting the API) need the ADMIN_PASSWORD secret. AWS settings and credentials come from
Streamlit secrets or the environment, never from the code.
"""

import hmac
import io
import os

import numpy as np
import pandas as pd
import streamlit as st

from spendcoder import data, metrics
from spendcoder.baseline import TfidfBaseline
from spendcoder.predictor import Predictor
from spendcoder.taxonomy import CATEGORIES, LABEL2ID

st.set_page_config(page_title="SpendCoder", page_icon="🧾", layout="wide")
REPO = os.path.dirname(os.path.abspath(__file__))
COLAB = "https://colab.research.google.com/github/saeedm31/SpendCoder/blob/main/notebooks/SpendCoder_finetuning.ipynb"
EXAMPLES = """Annual maintenance contract for production line pumps
Dell Latitude laptops and docking stations
Office cleaning and security guard services
Consulting support for SAP S/4HANA finance migration
Asbestos removal and soil remediation at warehouse site
Replacement electrical transformer and cabling
Gas chromatograph and lab consumables"""


def setting(name, default=None):
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default)


AWS = {k: setting(k) for k in ("AWS_REGION", "SPENDCODER_BUCKET", "SAGEMAKER_ROLE_ARN", "SAGEMAKER_IMAGE_URI",
                               "ECS_CLUSTER", "ECS_SERVICE", "API_URL")}
AWS_READY = bool(AWS["SPENDCODER_BUCKET"] and AWS["SAGEMAKER_ROLE_ARN"])
MODEL_REPO = setting("MODEL_REPO", "Saeedhoj/spendcoder-distilroberta")


def is_admin():
    secret = setting("ADMIN_PASSWORD")
    if not secret:
        return False
    return hmac.compare_digest(st.session_state.get("pw", ""), secret)


def client(name):
    import boto3
    return boto3.client(name, region_name=AWS["AWS_REGION"] or "eu-west-1")


@st.cache_data(show_spinner="Downloading contracts from USAspending.gov...", ttl=24 * 3600)
def public_data(per_year):
    return data.build_dataset(data.fetch(2018, 2025, per_year, log=lambda m: None))


@st.cache_resource(show_spinner="Loading the fine-tuned model...")
def load_model(source, repo):
    if source == "AWS S3 (promoted model)":
        from spendcoder import aws
        return Predictor.load(aws.download_active(client("s3"), AWS["SPENDCODER_BUCKET"], "/tmp/spendcoder-ui-model"))
    return Predictor.from_hub(repo)


def get_model():
    try:
        return load_model(st.session_state.get("source", "Hugging Face Hub"), MODEL_REPO), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---------------- sidebar ----------------
st.sidebar.title("🧾 SpendCoder")
page = st.sidebar.radio("Process", ["Overview", "1 · Data", "2 · Train", "3 · Evaluate", "4 · Classify",
                                    "5 · Review queue", "6 · Deploy"])
st.sidebar.selectbox("Model used for predictions", ["Hugging Face Hub"] + (["AWS S3 (promoted model)"] if AWS_READY else []),
                     key="source", help=f"Hub repo: {MODEL_REPO}")
with st.sidebar.expander("Admin"):
    st.text_input("Password", type="password", key="pw",
                  help="Needed for actions that cost money or change production.")
    st.caption("✅ admin" if is_admin() else "Read-only demo mode")
st.sidebar.caption(f"AWS: {'connected settings' if AWS_READY else 'not configured (optional)'}")
st.sidebar.caption("[Code and docs](https://github.com/saeedm31/SpendCoder)")


# ---------------- pages ----------------
if page == "Overview":
    st.title("SpendCoder: purchase lines to spend categories")
    st.write("A company's accounting team gets thousands of purchase and invoice lines. Each needs the right **spend "
             "category** before it is posted or analysed. SpendCoder is a **fine-tuned transformer** (DistilRoBERTa, "
             "Hugging Face) that suggests the category, says **how sure** it is, and sends unsure lines to a **person**. "
             "This panel runs the whole process; use the steps on the left.")
    st.graphviz_chart("""digraph {
      rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef6f5", fontname="sans-serif"];
      data [label="1 Data\\nreal purchase lines\\n+ official codes"]; base [label="2a Baseline\\nTF-IDF + logistic reg."];
      ft [label="2b Fine-tune\\nDistilRoBERTa\\nColab / SageMaker / local"]; ev [label="3 Evaluate\\nnewest year,\\nthreshold"];
      hub [label="Model registry\\nHF Hub / S3 versions"]; api [label="6 Deploy\\nFastAPI on\\nECS Fargate"];
      cls [label="4 Classify\\nauto if confident"]; rev [label="5 Review\\npeople correct\\nunsure lines"];
      data -> base -> ev; data -> ft -> ev; ft -> hub -> api -> cls -> rev; rev -> data [style=dashed, label="new labels"];
    }""")
    c = st.columns(3)
    c[0].metric("Dataset in this session", f"{len(st.session_state['df']):,} lines" if "df" in st.session_state else "not loaded")
    c[1].metric("Baseline", f"{st.session_state['baseline']['macro_f1']:.3f} macro F1" if "baseline" in st.session_state else "not trained")
    model, err = get_model()
    c[2].metric("Fine-tuned model", "available" if model else "not published yet")

elif page == "1 · Data":
    st.title("1 · Data")
    st.write("**Input:** a short purchase description. **Label:** its spend category. Use the public data, or upload "
             "your own company's labelled lines to see how it works on your wording.")
    src = st.radio("Source", ["Public: US Department of Energy contracts (USAspending.gov)",
                              "Upload my own CSV (columns: text, category)"], horizontal=True)
    if src.startswith("Public"):
        limit = 8000 if is_admin() else 1500
        per_year = st.slider("Contracts per year (2018-2025)", 500, limit, min(1500, limit), 500,
                             help="Larger downloads need admin. The full notebook run uses 8,000 per year.")
        if st.button("Load data", type="primary"):
            st.session_state["df"] = public_data(per_year)
            st.session_state["df_source"] = "public"
    else:
        up = st.file_uploader("CSV with columns text, category (category must be one of the 14 below)", type="csv")
        st.caption(", ".join(CATEGORIES))
        if up is not None:
            own = pd.read_csv(up)
            own = own[own["category"].isin(LABEL2ID)].assign(
                text=lambda d: d["text"].map(data.clean_text), label=lambda d: d["category"].map(LABEL2ID),
                start_date=pd.Timestamp("2020-01-01"))
            st.session_state["df"], st.session_state["df_source"] = own, "upload"
    if "df" in st.session_state:
        df = st.session_state["df"]
        if st.session_state.get("df_source") == "public":
            train, val, test = data.split(df)
            st.success(f"{len(df):,} lines · train {len(train):,} · validation {len(val):,} · test {len(test):,} "
                       "(split by contract start date: past to train, newest year to test)")
            st.bar_chart(data.summary(train, val, test))
        else:
            st.success(f"{len(df):,} of your lines loaded")
            st.bar_chart(df["category"].value_counts())
        st.dataframe(df.sample(min(15, len(df)), random_state=1)[["text", "category"]], hide_index=True, width="stretch")
        st.caption("Each row is one purchase line, not a customer.")

elif page == "2 · Train":
    st.title("2 · Train")
    st.subheader("a) Baseline: TF-IDF + logistic regression (runs here, seconds)")
    st.write("Always the first model: cheap, fast, explainable. The fine-tuned transformer has to beat it.")
    if "df" not in st.session_state:
        st.info("Load data first (step 1).")
    elif st.button("Train baseline", type="primary"):
        df = st.session_state["df"]
        if st.session_state.get("df_source") == "public":
            train, _, test = data.split(df)
        else:
            idx = np.random.default_rng(0).permutation(len(df))
            cut = int(0.8 * len(df))
            train, test = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
        base = TfidfBaseline().fit(train["text"], train["label"])
        st.session_state["baseline"] = metrics.scores(test["label"], base.predict_proba(test["text"]))
    if "baseline" in st.session_state:
        b = st.session_state["baseline"]
        c = st.columns(3)
        c[0].metric("Accuracy", f"{b['accuracy']:.3f}")
        c[1].metric("Macro F1", f"{b['macro_f1']:.3f}", help="Average over categories: small ones count as much as big ones")
        c[2].metric("Weighted F1", f"{b['weighted_f1']:.3f}")

    st.subheader("b) Fine-tune DistilRoBERTa (needs a GPU)")
    colab, sage, local = st.tabs(["Google Colab (free GPU)", "AWS SageMaker", "Local / your own GPU"])
    with colab:
        st.write("Step-by-step notebook with explanations: data, baseline, tokenizer, model, training arguments, "
                 "training, evaluation, threshold, ONNX export, Hub upload. About 30 minutes on a free T4.")
        st.link_button("Open the notebook in Colab", COLAB)
    with sage:
        st.write("Runs `scripts/train.py` as a SageMaker training job on `ml.g4dn.xlarge` (T4 GPU). Spot capacity "
                 "cuts the cost by up to about 70%. The model and its ONNX export land in S3.")
        if not AWS_READY:
            st.info("AWS is optional and not configured. Create the resources with `infra/terraform`, then set "
                    "SPENDCODER_BUCKET, SAGEMAKER_ROLE_ARN, AWS_REGION and AWS credentials in Streamlit secrets "
                    "(see docs/aws.md).")
        else:
            from spendcoder import aws
            c = st.columns(3)
            epochs = c[0].number_input("Epochs", 1, 12, 8)
            itype = c[1].selectbox("Instance", ["ml.g4dn.xlarge", "ml.g5.xlarge", "ml.p3.2xlarge"])
            spot = c[2].checkbox("Use spot capacity", True)
            if st.button("Launch training job", type="primary", disabled=not is_admin() or "df" not in st.session_state,
                         help="Needs admin and loaded public data"):
                buf = io.BytesIO()
                st.session_state["df"].to_csv(buf, index=False, compression="gzip")
                client("s3").put_object(Bucket=AWS["SPENDCODER_BUCKET"], Key=aws.DATA_KEY, Body=buf.getvalue())
                job = aws.launch_training(client("s3"), client("sagemaker"), AWS["SPENDCODER_BUCKET"],
                                          AWS["SAGEMAKER_ROLE_ARN"], REPO, AWS["AWS_REGION"] or "eu-west-1",
                                          image_uri=AWS["SAGEMAKER_IMAGE_URI"], instance_type=itype,
                                          epochs=int(epochs), spot=spot)
                st.success(f"Started {job}")
            if st.button("Refresh job list"):
                pass
            try:
                st.dataframe(pd.DataFrame(aws.list_jobs(client("sagemaker"))), hide_index=True, width="stretch")
            except Exception as e:
                st.warning(f"Could not list jobs: {e}")
    with local:
        st.code("pip install -r requirements-train.txt\npython scripts/prepare_data.py\n"
                "python scripts/train.py --push <hf-user>/spendcoder-distilroberta", language="bash")

elif page == "3 · Evaluate":
    st.title("3 · Evaluate the fine-tuned model")
    model, err = get_model()
    if not model:
        st.info(f"No fine-tuned model available yet ({err}). Train one in step 2.")
    else:
        if model.meta:
            rows = {k.replace("_test", "").replace("finetuned", "fine-tuned"): v for k, v in model.meta.items()}
            st.write("**Scores recorded at training time (test year):**")
            st.dataframe(pd.DataFrame(rows).T.round(3), width="stretch")
        if st.session_state.get("df_source") != "public":
            st.info("Load the public data (step 1) to re-score the model here on its test year.")
        elif st.button("Score the test year now", type="primary"):
            _, _, test = data.split(st.session_state["df"])
            test = test.head(3000)
            p = model.probs(test["text"].tolist())
            st.session_state["eval"] = (test, p)
        if "eval" in st.session_state:
            test, p = st.session_state["eval"]
            s = metrics.scores(test["label"], p)
            c = st.columns(3)
            c[0].metric("Accuracy", f"{s['accuracy']:.3f}")
            c[1].metric("Macro F1", f"{s['macro_f1']:.3f}",
                        delta=(f"{s['macro_f1'] - st.session_state['baseline']['macro_f1']:+.3f} vs baseline"
                               if "baseline" in st.session_state else None))
            c[2].metric("Test lines scored", f"{len(test):,}")
            st.subheader("Automation vs accuracy")
            curve = pd.DataFrame(metrics.automation(test["label"], p, thresholds=np.round(np.arange(0.3, 1.0, 0.05), 2)))
            st.line_chart(curve.set_index("threshold")[["automated_pct", "accuracy_of_automated_pct"]])
            st.caption("Higher threshold: fewer lines coded automatically, but those are more accurate. Pick the point "
                       "your accounting team accepts; the rest go to the review queue.")
            st.subheader("Per category (F1)")
            rep = pd.DataFrame(metrics.report(test["label"], p)).T.loc[CATEGORIES]
            st.bar_chart(rep["f1-score"])
            st.subheader("Most confident mistakes")
            view = test.assign(predicted=[CATEGORIES[i] for i in p.argmax(1)], confidence=p.max(1))
            st.dataframe(view[view["predicted"] != view["category"]].sort_values("confidence", ascending=False)
                         [["text", "category", "predicted", "confidence"]].head(20), hide_index=True, width="stretch")

elif page == "4 · Classify":
    st.title("4 · Classify purchase lines")
    model, err = get_model()
    if not model:
        st.info(f"No fine-tuned model available yet ({err}).")
    else:
        threshold = st.slider("Review threshold: below this confidence a person checks the line", 0.30, 0.99,
                              float(model.threshold), 0.01)
        text = st.text_area("One purchase line per row", EXAMPLES, height=180)
        up = st.file_uploader("...or a CSV with a 'text' column", type="csv")
        lines = (pd.read_csv(up)["text"].astype(str).tolist() if up is not None
                 else [line.strip() for line in text.splitlines() if line.strip()])[:2000]
        if lines:
            model.threshold = threshold
            res = model.classify(lines)
            out = pd.DataFrame({"text": [r["text"] for r in res], "category": [r["category"] for r in res],
                                "confidence": [r["confidence"] for r in res],
                                "decision": ["person checks" if r["needs_review"] else "automatic" for r in res],
                                "second_choice": [r["suggestions"][1]["category"] for r in res]})
            st.session_state["queue"] = out[out["decision"] == "person checks"].reset_index(drop=True)
            st.dataframe(out, hide_index=True, width="stretch", column_config={
                "confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
            st.caption(f"Each row is one purchase line. {int((out['decision'] == 'automatic').sum())} of {len(out)} "
                       f"would be coded automatically; {len(st.session_state['queue'])} go to the review queue (step 5).")
            st.download_button("Download results (CSV)", out.to_csv(index=False), "spendcoder_results.csv")

elif page == "5 · Review queue":
    st.title("5 · Review queue")
    st.write("Lines the model was unsure about. A person picks the right category; the corrections become training "
             "data for the next fine-tuning round.")
    q = st.session_state.get("queue")
    if q is None or q.empty:
        st.info("Nothing to review. Classify some lines in step 4.")
    else:
        edited = st.data_editor(q.assign(correct_category=q["category"]), hide_index=True, width="stretch",
                                disabled=["text", "category", "confidence", "decision", "second_choice"],
                                column_config={"correct_category": st.column_config.SelectboxColumn(
                                    "Correct category", options=CATEGORIES, required=True)})
        corrections = edited[["text", "correct_category"]].rename(columns={"correct_category": "category"})
        st.download_button("Download corrections (CSV)", corrections.to_csv(index=False), "spendcoder_corrections.csv")
        if AWS_READY and st.button("Save corrections to S3", disabled=not is_admin()):
            from spendcoder import aws
            st.success("Saved " + aws.save_reviews(client("s3"), AWS["SPENDCODER_BUCKET"],
                                                   corrections.to_csv(index=False).encode()))

elif page == "6 · Deploy":
    st.title("6 · Deploy")
    st.write("The API (`serve/app.py`) is a FastAPI service in Docker that runs the ONNX model on CPU. Options: "
             "local Docker, any container platform, or AWS ECS Fargate behind a load balancer (`infra/terraform`).")
    st.code("docker build -t spendcoder .\ndocker run -e MODEL_REPO=<hf-user>/spendcoder-distilroberta -p 8000:8000 spendcoder",
            language="bash")
    if not AWS_READY:
        st.info("AWS is optional and not configured. See docs/aws.md for the Terraform setup.")
    else:
        from spendcoder import aws
        s3, sm = client("s3"), client("sagemaker")
        st.subheader("Model in production")
        st.json(aws.active_model(s3, AWS["SPENDCODER_BUCKET"]) or {"active": "none yet"})
        done = [j["name"] for j in aws.list_jobs(sm, 20) if j["status"] == "Completed"]
        job = st.selectbox("Completed training jobs", done) if done else None
        c = st.columns(2)
        if c[0].button("Promote to production", disabled=not (is_admin() and job),
                       help="Copies the job's ONNX model to models/active/ (previous versions stay in S3)"):
            st.success(aws.promote(s3, sm, AWS["SPENDCODER_BUCKET"], job))
        if c[1].button("Restart API to load it", disabled=not (is_admin() and AWS["ECS_CLUSTER"])):
            st.success(aws.restart_service(client("ecs"), AWS["ECS_CLUSTER"], AWS["ECS_SERVICE"]))
        if AWS["API_URL"] and st.button("Check API health"):
            import requests
            try:
                st.json(requests.get(AWS["API_URL"].rstrip("/") + "/health", timeout=10).json())
            except Exception as e:
                st.error(f"API not reachable: {e}")
