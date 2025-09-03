import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="AI Mastery Roadmap Tracker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        color: #666;
        margin-bottom: 2rem;
    }
    .level-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid;
    }
    .junior-card {
        border-left-color: #4CAF50;
        background: linear-gradient(135deg, #f1f8e9, #e8f5e9);
    }
    .mid-card {
        border-left-color: #FFC107;
        background: linear-gradient(135deg, #fffde7, #fff9c4);
    }
    .senior-card {
        border-left-color: #F44336;
        background: linear-gradient(135deg, #ffebee, #ffcdd2);
    }
    .progress-bar {
        height: 20px;
        border-radius: 10px;
        background-color: #e0e0e0;
        margin: 10px 0;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .resource-card {
        background-color: #e3f2fd;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Data directory
DATA_DIR = Path("data")
ROADMAP_FILE = DATA_DIR / "roadmap_progress.csv"
TASKS_FILE = DATA_DIR / "tasks.csv"

# Load data
@st.cache_data
def load_data():
    if not ROADMAP_FILE.exists():
        st.error("Roadmap data not found. Please run create_data_structure.py first.")
        st.stop()
    
    roadmap_df = pd.read_csv(ROADMAP_FILE)
    tasks_df = pd.read_csv(TASKS_FILE) if TASKS_FILE.exists() else pd.DataFrame()
    return roadmap_df, tasks_df

def save_data(roadmap_df, tasks_df):
    roadmap_df.to_csv(ROADMAP_FILE, index=False)
    if not tasks_df.empty:
        tasks_df.to_csv(TASKS_FILE, index=False)

# Initialize session state
if 'roadmap_df' not in st.session_state or 'tasks_df' not in st.session_state:
    roadmap_df, tasks_df = load_data()
    st.session_state.roadmap_df = roadmap_df
    st.session_state.tasks_df = tasks_df

# Header
st.markdown('<div class="main-header">🧠 AI Mastery Roadmap Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Track your progress through Junior, Mid, and Senior levels of AI mastery</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Settings")
    level_filter = st.selectbox(
        "Filter by Level",
        ["All", "Junior", "Mid", "Senior"]
    )
    
    show_resources = st.checkbox("Show Learning Resources", value=True)
    show_details = st.checkbox("Show Detailed Progress", value=True)
    
    # Overall progress
    st.header("Overall Progress")
    total_topics = st.session_state.roadmap_df['topics_total'].sum()
    completed_topics = st.session_state.roadmap_df['topics_completed'].sum()
    progress_percent = (completed_topics / total_topics * 100) if total_topics > 0 else 0
    
    st.metric("Topics Completed", f"{completed_topics}/{total_topics}")
    st.progress(progress_percent / 100)
    
    # Level progress
    level_progress = st.session_state.roadmap_df.groupby('level').agg({
        'topics_total': 'sum',
        'topics_completed': 'sum'
    }).reset_index()
    
    for _, row in level_progress.iterrows():
        level_pct = (row['topics_completed'] / row['topics_total'] * 100) if row['topics_total'] > 0 else 0
        st.write(f"{row['level']}: {row['topics_completed']}/{row['topics_total']} ({level_pct:.1f}%)")

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Progress Overview", "📝 Task Manager", "📚 Learning Resources"])

with tab1:
    # Filter data based on selection
    filtered_df = st.session_state.roadmap_df
    if level_filter != "All":
        filtered_df = filtered_df[filtered_df['level'] == level_filter]
    
    # Display progress by level
    for level in filtered_df['level'].unique():
        level_df = filtered_df[filtered_df['level'] == level]
        
        # Level header
        level_title = f"{level} Level"
        if level == "Junior":
            st.markdown(f'<div class="level-card junior-card"><h2>🟢 {level_title}</h2></div>', unsafe_allow_html=True)
        elif level == "Mid":
            st.markdown(f'<div class="level-card mid-card"><h2>🟡 {level_title}</h2></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="level-card senior-card"><h2>🔴 {level_title}</h2></div>', unsafe_allow_html=True)
        
        # Display categories
        for _, row in level_df.iterrows():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(row['category'])
                if pd.notna(row['description']):
                    st.caption(row['description'])
                
                if show_details:
                    # Progress bars
                    topics_pct = (row['topics_completed'] / row['topics_total'] * 100) if row['topics_total'] > 0 else 0
                    tasks_pct = (row['tasks_completed'] / row['tasks_total'] * 100) if row['tasks_total'] > 0 else 0
                    
                    st.write(f"Topics: {row['topics_completed']}/{row['topics_total']} ({topics_pct:.1f}%)")
                    st.progress(topics_pct / 100)
                    
                    st.write(f"Tasks: {row['tasks_completed']}/{row['tasks_total']} ({tasks_pct:.1f}%)")
                    st.progress(tasks_pct / 100)
            
            with col2:
                # Update buttons
                st.write("Update Progress")
                
                # Topics counter
                topics_count = st.number_input(
                    f"Topics completed for {row['category']}",
                    min_value=0,
                    max_value=row['topics_total'],
                    value=row['topics_completed'],
                    key=f"topics_{row['category']}"
                )
                
                # Tasks counter
                tasks_count = st.number_input(
                    f"Tasks completed for {row['category']}",
                    min_value=0,
                    max_value=row['tasks_total'],
                    value=row['tasks_completed'],
                    key=f"tasks_{row['category']}"
                )
                
                # Update button
                if st.button("Update", key=f"update_{row['category']}"):
                    st.session_state.roadmap_df.loc[
                        (st.session_state.roadmap_df['level'] == row['level']) & 
                        (st.session_state.roadmap_df['category'] == row['category']),
                        'topics_completed'
                    ] = topics_count
                    
                    st.session_state.roadmap_df.loc[
                        (st.session_state.roadmap_df['level'] == row['level']) & 
                        (st.session_state.roadmap_df['category'] == row['category']),
                        'tasks_completed'
                    ] = tasks_count
                    
                    save_data(st.session_state.roadmap_df, st.session_state.tasks_df)
                    st.success("Progress updated!")
                    st.rerun()
        
        st.divider()

with tab2:
    st.header("Task Manager")
    
    if st.session_state.tasks_df.empty:
        st.info("No tasks loaded. Run create_data_structure.py to generate tasks.")
    else:
        # Filter tasks
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
            # Get categories for the selected level
            level_categories = st.session_state.roadmap_df[
                st.session_state.roadmap_df['level'] == task_level_filter
            ]['category'].unique()
            filtered_tasks = filtered_tasks[filtered_tasks['category'].isin(level_categories)]
        
        if task_category_filter != "All":
            filtered_tasks = filtered_tasks[filtered_tasks['category'] == task_category_filter]
        
        # Display tasks
        for _, task in filtered_tasks.iterrows():
            col1, col2 = st.columns([5, 1])
            with col1:
                status = "✅" if task['completed'] else "⬜"
                st.write(f"{status} {task['task']}")
                st.caption(f"Category: {task['category']}")
            with col2:
                new_status = st.checkbox(
                    "Completed", 
                    value=task['completed'], 
                    key=f"task_{task['task']}"
                )
                if new_status != task['completed']:
                    st.session_state.tasks_df.loc[
                        st.session_state.tasks_df['task'] == task['task'],
                        'completed'
                    ] = new_status
                    save_data(st.session_state.roadmap_df, st.session_state.tasks_df)
                    st.rerun()

with tab3:
    st.header("Learning Resources")
    
    if show_resources:
        for level in ["Junior", "Mid", "Senior"]:
            st.subheader(f"{level} Level Resources")
            level_df = st.session_state.roadmap_df[st.session_state.roadmap_df['level'] == level]
            
            for _, row in level_df.iterrows():
                with st.expander(row['category']):
                    if pd.notna(row['resources']):
                        resources = row['resources'].split(', ')
                        for resource in resources:
                            st.markdown(f'<div class="resource-card">📚 {resource}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No resources listed for this category.")

# Footer
st.divider()
st.caption("AI Mastery Roadmap Tracker • Built with Streamlit • Data is stored locally in CSV files")