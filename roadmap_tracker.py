# roadmap_tracker.py
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI Mastery Roadmap Tracker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f3a60;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #374151; /* higher contrast */
        margin-bottom: 2rem;
    }

    /* ---- Level Banner (polished) ---- */
    .level-banner {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 16px 18px;
        border-radius: 14px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.07);
        margin: 18px 0 12px 0;
    }
    .level-banner .level-icon {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        border-radius: 999px;
        font-size: 22px;
        background: rgba(255,255,255,0.7);
        box-shadow: inset 0 0 0 2px rgba(255,255,255,0.6);
    }
    .level-banner .level-title {
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: .2px;
        color: #0f172a; /* slate-900 */
        margin: 0;
    }
    .level-banner .level-chip {
        margin-left: auto;
        padding: 6px 10px;
        font-size: 0.85rem;
        font-weight: 700;
        color: #0f172a;
        border-radius: 999px;
        background: linear-gradient(135deg, rgba(255,255,255,0.85), rgba(255,255,255,0.6));
        border: 1px solid rgba(0,0,0,0.06);
    }

    /* Color themes per level */
    .level-junior {
        background: linear-gradient(135deg, #e8f7ec, #dff5e7);
        border-left: 6px solid #22c55e; /* green-500 */
    }
    .level-mid {
        background: linear-gradient(135deg, #fff7db, #fff0b3);
        border-left: 6px solid #f59e0b; /* amber-500 */
    }
    .level-senior {
        background: linear-gradient(135deg, #ffe3e3, #ffc9cc);
        border-left: 6px solid #ef4444; /* red-500 */
    }

    /* ---- Cards (keep subtle) ---- */
    .level-card {
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 0.25rem 0 1rem 0;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.06);
        border-left: 4px solid rgba(0,0,0,0.06);
        color: #0f172a;
    }

    /* ---- High-contrast progress labels as pills ---- */
    .stat-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 8px 0 4px 0;
    }
    .stat-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border-radius: 999px;
        font-weight: 700;
        letter-spacing: .2px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.06);
        color: #0b1220;            /* text color */
        background: #eef2f7;       /* pill bg */
    }
    .stat-pill .label {
        opacity: .85;
        font-weight: 600;
    }
    .stat-pill .value {
        font-variant-numeric: tabular-nums;
        font-weight: 800;
    }

    /* Make Streamlit progress track slightly darker for visibility */
    [data-testid="stProgress"] > div {
        background-color: #d7dce1 !important;
        border-radius: 999px;
    }
    [data-testid="stProgress"] > div > div {
        border-radius: 999px;
    }

    /* Resource cards */
    .resource-card {
        background-color: #ffffff;
        border: 2px solid #2196F3;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(33, 150, 243, 0.15);
        transition: all 0.3s ease;
    }
    .resource-card:hover {
        box-shadow: 0 4px 12px rgba(33, 150, 243, 0.25);
        transform: translateY(-2px);
    }
    .resource-link {
        color: #1976D2;
        text-decoration: none;
        font-weight: 500;
        font-size: 1rem;
    }
    .resource-link:hover {
        color: #0D47A1;
        text-decoration: underline;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 16px;
        margin: 16px 0;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Data paths ----------
DATA_DIR = Path("data")
ROADMAP_FILE = DATA_DIR / "roadmap_progress.csv"
TASKS_FILE = DATA_DIR / "tasks.csv"

# ---------- Data I/O ----------
@st.cache_data
def load_data():
    default_roadmap = pd.DataFrame({
        'level': [], 'category': [], 'topics_total': [], 'topics_completed': [],
        'tasks_total': [], 'tasks_completed': [], 'resources': [], 'description': []
    })
    default_tasks = pd.DataFrame({'category': [], 'task': [], 'completed': []})

    if not ROADMAP_FILE.exists():
        return default_roadmap, default_tasks

    try:
        roadmap_df = pd.read_csv(ROADMAP_FILE)
        tasks_df = pd.read_csv(TASKS_FILE) if TASKS_FILE.exists() else default_tasks
        # Normalize boolean for 'completed' if coming as 0/1 or strings
        if not tasks_df.empty:
            tasks_df['completed'] = tasks_df['completed'].astype(str).str.lower().isin(['1', 'true', 'yes'])
        return roadmap_df, tasks_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return default_roadmap, default_tasks

def save_data(roadmap_df, tasks_df):
    DATA_DIR.mkdir(exist_ok=True)
    roadmap_df.to_csv(ROADMAP_FILE, index=False)
    if not tasks_df.empty:
        tasks_df_to_save = tasks_df.copy()
        tasks_df_to_save['completed'] = tasks_df_to_save['completed'].astype(int)
        tasks_df_to_save.to_csv(TASKS_FILE, index=False)

def get_category_tasks(tasks_df, category):
    if tasks_df.empty:
        return pd.DataFrame(columns=['category', 'task', 'completed'])
    return tasks_df[tasks_df['category'] == category]

def update_roadmap_task_counts(roadmap_df, tasks_df):
    for idx, row in roadmap_df.iterrows():
        category_tasks = get_category_tasks(tasks_df, row['category'])
        if not category_tasks.empty:
            completed_count = int(category_tasks['completed'].sum())
            roadmap_df.at[idx, 'tasks_completed'] = completed_count
            roadmap_df.at[idx, 'tasks_total'] = len(category_tasks)
        else:
            roadmap_df.at[idx, 'tasks_completed'] = 0
            roadmap_df.at[idx, 'tasks_total'] = 0
    return roadmap_df

# ---------- Session init ----------
if 'roadmap_df' not in st.session_state or 'tasks_df' not in st.session_state:
    roadmap_df, tasks_df = load_data()
    st.session_state.roadmap_df = roadmap_df
    st.session_state.tasks_df = tasks_df

# ---------- Header ----------
st.markdown('<div class="main-header">🧠 AI Mastery Roadmap Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Track your progress through Junior, Mid, and Senior levels of AI mastery</div>', unsafe_allow_html=True)

# ---------- Empty data guard ----------
if st.session_state.roadmap_df.empty:
    st.markdown(
        '<div class="warning-box">'
        '<h3>⚠️ No Data Found</h3>'
        '<p>Please run the data creation script first:</p>'
        '<code>python create_data.py</code>'
        '</div>',
        unsafe_allow_html=True
    )
    st.stop()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Settings")
    level_filter = st.selectbox("Filter by Level", ["All", "Junior", "Mid", "Senior"])

    show_resources = st.checkbox("Show Learning Resources", value=True)
    show_details = st.checkbox("Show Detailed Progress", value=True)

    st.header("Overall Progress")
    total_topics = int(st.session_state.roadmap_df['topics_total'].sum())
    completed_topics = int(st.session_state.roadmap_df['topics_completed'].sum())
    progress_percent = (completed_topics / total_topics * 100) if total_topics > 0 else 0

    st.metric("Topics Completed", f"{completed_topics}/{total_topics}")
    st.progress(progress_percent / 100)

    level_progress = st.session_state.roadmap_df.groupby('level').agg({
        'topics_total': 'sum',
        'topics_completed': 'sum'
    }).reset_index()

    for _, row in level_progress.iterrows():
        lvl = row['level']
        done = int(row['topics_completed'])
        tot = int(row['topics_total'])
        level_pct = (done / tot * 100) if tot > 0 else 0
        st.write(f"{lvl}: {done}/{tot} ({level_pct:.1f}%)")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📊 Progress Overview", "📝 Task Manager", "📚 Learning Resources"])

with tab1:
    # Filter by level if selected
    filtered_df = st.session_state.roadmap_df
    if level_filter != "All":
        filtered_df = filtered_df[filtered_df['level'] == level_filter]

    # Render per level
    for level in filtered_df['level'].unique():
        level_df = filtered_df[filtered_df['level'] == level]

        # Polished banner (instead of plain "🟢 Junior Level")
        if level == "Junior":
            banner_class = "level-banner level-junior"
            icon = "🟢"
        elif level == "Mid":
            banner_class = "level-banner level-mid"
            icon = "🟡"
        else:
            banner_class = "level-banner level-senior"
            icon = "🔴"

        st.markdown(
            f'''
            <div class="{banner_class}">
                <div class="level-icon">{icon}</div>
                <div class="level-title">{level} Level</div>
                <div class="level-chip">Roadmap Section</div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        # Optional subtle card for a bit of separation
        st.markdown('<div class="level-card">', unsafe_allow_html=True)

        # Categories within the level
        for _, row in level_df.iterrows():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.subheader(row['category'])
                if pd.notna(row.get('description', None)) and str(row['description']).strip():
                    st.caption(row['description'])

                if show_details:
                    # Topics progress
                    topics_total = int(row['topics_total'])
                    topics_done = int(row['topics_completed'])
                    topics_pct = (topics_done / topics_total * 100) if topics_total > 0 else 0

                    # Tasks progress based on actual task table
                    category_tasks = get_category_tasks(st.session_state.tasks_df, row['category'])
                    if not category_tasks.empty:
                        tasks_done = int(category_tasks['completed'].sum())
                        tasks_total = int(len(category_tasks))
                    else:
                        tasks_done = int(row.get('tasks_completed', 0) or 0)
                        tasks_total = int(row.get('tasks_total', 0) or 0)
                    tasks_pct = (tasks_done / tasks_total * 100) if tasks_total > 0 else 0

                    # ---- High-contrast pills for labels ----
                    st.markdown(
                        f'''
                        <div class="stat-row">
                            <div class="stat-pill">
                                <span class="label">Topics</span>
                                <span class="value">{topics_done}/{topics_total} ({topics_pct:.1f}%)</span>
                            </div>
                            <div class="stat-pill">
                                <span class="label">Tasks</span>
                                <span class="value">{tasks_done}/{tasks_total} ({tasks_pct:.1f}%)</span>
                            </div>
                        </div>
                        ''',
                        unsafe_allow_html=True
                    )

                    st.progress(topics_pct / 100)
                    st.progress(tasks_pct / 100)

                    # --- Dropdown (expander) listing all milestone tasks with checkboxes ---
                    if not category_tasks.empty:
                        with st.expander("📋 Tasks for this milestone"):
                            completed_count = int(category_tasks['completed'].sum())
                            st.write(f"Progress: {completed_count}/{len(category_tasks)} tasks completed")
                            st.progress(completed_count / len(category_tasks) if len(category_tasks) > 0 else 0)

                            key_prefix = f"task_{row['category']}".replace(" ", "_").lower()
                            for task_idx, task_row in category_tasks.iterrows():
                                label_icon = "✅" if task_row['completed'] else "❌"
                                new_status = st.checkbox(
                                    f"{label_icon} {task_row['task']}",
                                    value=bool(task_row['completed']),
                                    key=f"{key_prefix}_{task_idx}"
                                )
                                if new_status != bool(task_row['completed']):
                                    st.session_state.tasks_df.at[task_idx, 'completed'] = bool(new_status)
                                    st.session_state.roadmap_df = update_roadmap_task_counts(
                                        st.session_state.roadmap_df,
                                        st.session_state.tasks_df
                                    )
                                    save_data(st.session_state.roadmap_df, st.session_state.tasks_df)
                                    st.rerun()

            with col2:
                # Topics progress editor (tasks handled in dropdown)
                st.write("**Update Topics Progress**")
                topics_count = st.number_input(
                    f"Topics completed",
                    min_value=0,
                    max_value=int(row['topics_total']),
                    value=int(row['topics_completed']),
                    key=f"topics_{row['category']}_{level}"
                )

                if st.button("Update Topics", key=f"update_topics_{row['category']}_{level}"):
                    st.session_state.roadmap_df.loc[
                        (st.session_state.roadmap_df['level'] == row['level']) &
                        (st.session_state.roadmap_df['category'] == row['category']),
                        'topics_completed'
                    ] = int(topics_count)

                    save_data(st.session_state.roadmap_df, st.session_state.tasks_df)
                    st.success("Topics progress updated!")
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)  # close level-card
        st.divider()

with tab2:
    st.header("Task Manager")
    if st.session_state.tasks_df.empty:
        st.info("No tasks loaded. Run create_data.py to generate tasks.")
    else:
        task_level_filter = st.selectbox(
            "Filter tasks by level",
            ["All", "Junior", "Mid", "Senior"],
            key="task_level_filter"
        )
        task_category_filter = st.selectbox(
            "Filter by category",
            ["All"] + list(st.session_state.roadmap_df['category'].unique()),
            key="task_category_filter"
        )

        filtered_tasks = st.session_state.tasks_df
        if task_level_filter != "All":
            level_categories = st.session_state.roadmap_df[
                st.session_state.roadmap_df['level'] == task_level_filter
            ]['category'].unique()
            filtered_tasks = filtered_tasks[filtered_tasks['category'].isin(level_categories)]

        if task_category_filter != "All":
            filtered_tasks = filtered_tasks[filtered_tasks['category'] == task_category_filter]

        for category in filtered_tasks['category'].unique():
            category_tasks = filtered_tasks[filtered_tasks['category'] == category]
            with st.expander(f"📋 {category} ({len(category_tasks)} tasks)"):
                completed_count = int(category_tasks['completed'].sum())
                st.write(f"Progress: {completed_count}/{len(category_tasks)} tasks completed")
                st.progress(completed_count / len(category_tasks) if len(category_tasks) > 0 else 0)

                key_prefix = f"mgr_{category}".replace(" ", "_").lower()
                for task_idx, task in category_tasks.iterrows():
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        status = "✅" if task['completed'] else "❌"
                        st.write(f"{status} {task['task']}")
                    with col2:
                        new_status = st.checkbox(
                            "Done",
                            value=bool(task['completed']),
                            key=f"{key_prefix}_{task_idx}"
                        )
                        if new_status != bool(task['completed']):
                            st.session_state.tasks_df.at[task_idx, 'completed'] = bool(new_status)
                            st.session_state.roadmap_df = update_roadmap_task_counts(
                                st.session_state.roadmap_df,
                                st.session_state.tasks_df
                            )
                            save_data(st.session_state.roadmap_df, st.session_state.tasks_df)
                            st.rerun()

with tab3:
    st.header("Learning Resources")

    if show_resources:
        for level in ["Junior", "Mid", "Senior"]:
            st.subheader(f"{level} Level Resources")
            level_df = st.session_state.roadmap_df[st.session_state.roadmap_df['level'] == level]

            for _, row in level_df.iterrows():
                with st.expander(f"📚 {row['category']}"):
                    if pd.notna(row.get('resources', "")) and str(row['resources']).strip():
                        resources = [r.strip() for r in str(row['resources']).split(',')]

                        resource_links = {
                            "Coursera ML Course": "https://www.coursera.org/learn/machine-learning",
                            "Python for Data Science": "https://www.python.org/",
                            "Scikit-learn Documentation": "https://scikit-learn.org/",
                            "Kaggle Learn": "https://www.kaggle.com/learn",
                            "TensorFlow Tutorials": "https://www.tensorflow.org/tutorials",
                            "PyTorch Documentation": "https://pytorch.org/docs/",
                            "Deep Learning Specialization": "https://www.coursera.org/specializations/deep-learning",
                            "Fast.ai Course": "https://www.fast.ai/",
                            "Hugging Face Documentation": "https://huggingface.co/docs",
                            "spaCy Documentation": "https://spacy.io/",
                            "NLTK Book": "https://www.nltk.org/book/",
                            "OpenCV Documentation": "https://docs.opencv.org/",
                            "Computer Vision Course": "https://cs231n.github.io/",
                            "MLflow Documentation": "https://mlflow.org/docs/latest/index.html",
                            "Docker Documentation": "https://docs.docker.com/",
                            "Kubernetes Documentation": "https://kubernetes.io/docs/",
                            "Papers With Code": "https://paperswithcode.com/",
                            "arXiv.org": "https://arxiv.org/",
                            "Google AI Blog": "https://ai.googleblog.com/",
                            "OpenAI Blog": "https://openai.com/blog/"
                        }

                        for resource in resources:
                            resource_url = resource_links.get(
                                resource,
                                f"https://www.google.com/search?q={resource.replace(' ', '+')}"
                            )
                            st.markdown(f'''
                            <div class="resource-card">
                                📚 <a href="{resource_url}" target="_blank" class="resource-link">{resource}</a>
                                <br><small style="color: #666;">Click to access this resource</small>
                            </div>
                            ''', unsafe_allow_html=True)
                    else:
                        st.info("No resources listed for this category.")
                        st.write("**Suggested resources to add:**")
                        st.write("• Online courses and tutorials")
                        st.write("• Documentation and guides")
                        st.write("• Books and research papers")
                        st.write("• Practice projects and datasets")

# ---------- Footer ----------
st.divider()
st.caption("AI Mastery Roadmap Tracker • Built with Streamlit • Data is stored locally in CSV files")

if st.button("💾 Save All Progress", type="primary"):
    save_data(st.session_state.roadmap_df, st.session_state.tasks_df)
    st.success("All progress saved successfully!")

if ROADMAP_FILE.exists():
    last_modified = datetime.fromtimestamp(ROADMAP_FILE.stat().st_mtime)
    st.caption(f"Last updated: {last_modified.strftime('%Y-%m-%d %H:%M:%S')}")
