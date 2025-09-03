
import pandas as pd
from pathlib import Path

# Create data directory
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Define the roadmap structure
roadmap_data = {
    "level": ["Junior", "Junior", "Junior", "Junior", "Junior", "Junior", "Junior",
              "Mid", "Mid", "Mid", "Mid", "Mid", "Mid",
              "Senior", "Senior", "Senior", "Senior", "Senior", "Senior"],
    "category": ["Python Basics", "Pandas", "NumPy", "Matplotlib", "Math Foundations", 
                 "OOP Foundations", "Portfolio Projects",
                 "scikit-learn", "Feature Engineering", "Evaluation & Tuning", 
                 "Project Engineering", "Kaggle Track", "Mid-Level Overview",
                 "PyTorch", "CNNs", "RNNs & NLP", "Transformers", "GANs/SSL/RL", "MLOps"],
    "topics_total": [5, 6, 6, 5, 4, 4, 3, 7, 6, 6, 6, 5, 0, 6, 6, 6, 6, 6, 6],
    "topics_completed": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "tasks_total": [3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 0, 3, 4, 3, 3, 3, 3],
    "tasks_completed": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "resources": [
        "Python Crash Course, Automate the Boring Stuff, Corey Schafer YouTube",
        "Python for Data Analysis, freeCodeCamp Pandas Tutorial, Pandas Docs",
        "Numerical Python, NumPy in 1 Hour, NumPy User Guide",
        "Python Data Science Handbook, Corey Schafer Matplotlib, Matplotlib Gallery",
        "Mathematics for ML, Khan Academy, 3Blue1Brown",
        "Fluent Python, OOP in Python (freeCodeCamp)",
        "Kaggle Micro-courses, Distill.pub",
        "Hands-On ML, StatQuest ML Playlist, Scikit-learn User Guide",
        "Feature Engineering for ML, Kaggle FE Course, sklearn Feature Extraction",
        "Interpretable ML, StatQuest Metrics, sklearn Model Evaluation",
        "Effective Python, FastAPI Docs, Cookiecutter Data Science",
        "Kaggle Micro-courses, Kaggle Grandmaster writeups",
        "Hands-On ML, StatQuest YouTube, Effective Python, Kaggle Competitions, fast.ai",
        "Deep Learning with PyTorch, Daniel Bourke PyTorch Series, PyTorch Tutorials",
        "Deep Learning, Dive into Deep Learning, CS231n, TorchVision Models",
        "Speech and Language Processing, CS224N, The Annotated Transformer",
        "Natural Language Processing with Transformers, Yannic Kilcher, Hugging Face Course",
        "Generative Deep Learning, Reinforcement Learning, Lil'Log GAN series, OpenAI Spinning Up",
        "Building ML Pipelines, MLflow Docs, DVC Docs, Full Stack Deep Learning"
    ],
    "description": [
        "Control flow, functions, file handling, OOP basics, error handling",
        "DataFrame operations, indexing, missing values, GroupBy, merging",
        "Array operations, reshaping, math functions, random generation",
        "Plot types, subplots, annotations, figure customization",
        "Linear algebra, probability, calculus, distributions",
        "Classes, inheritance, magic methods, static vs class methods",
        "End-to-end projects, data cleaning toolkit, storytelling notebooks",
        "Train/test split, regression, SVMs, trees, forests, unsupervised learning",
        "Missing values, encoding, scaling, text processing, feature selection",
        "Metrics selection, cross-validation, hyperparameter tuning, model interpretation",
        "Repo structure, logging, testing, reproducibility, data versioning",
        "Titanic baseline, feature engineering, LB improvement, notebook reporting",
        "Core ML foundations, production-ready projects, Kaggle readiness",
        "Tensors, autograd, DataLoaders, training loops, optimization, checkpointing",
        "Conv/Pool fundamentals, CNNs, data augmentation, transfer learning, model export",
        "RNNs/GRU/LSTM, attention, tokenization, embeddings, model evaluation",
        "Self-attention, encoder blocks, Transformer training, pretrained models, deployment",
        "GANs, contrastive learning, reinforcement learning, evaluation, visualization",
        "FastAPI, Docker, MLflow, DVC, CI/CD, quantization, monitoring"
    ]
}

# Create DataFrame
df = pd.DataFrame(roadmap_data)

# Save to CSV
df.to_csv(data_dir / "roadmap_progress.csv", index=False)

# Create a separate table for tasks
tasks_data = []
for _, row in df.iterrows():
    if row['category'] == "Python Basics":
        tasks_data.extend([
            {"category": row['category'], "task": "CLI Utility → Build a small CLI that reads/writes files", "completed": False},
            {"category": row['category'], "task": "Refactor Script → Turn a script into reusable functions & modules", "completed": False},
            {"category": row['category'], "task": "OOP Mini-App → Create a class-based Todo manager with save/load", "completed": False}
        ])
    elif row['category'] == "Pandas":
        tasks_data.extend([
            {"category": row['category'], "task": "Sales Report → Analyze total sales by product & region", "completed": False},
            {"category": row['category'], "task": "Customer Segments → Group customers by behavior & spend", "completed": False},
            {"category": row['category'], "task": "Cleaning Notebook → Fix nulls/dupes and export a clean CSV", "completed": False}
        ])
    elif row['category'] == "NumPy":
        tasks_data.extend([
            {"category": row['category'], "task": "Crypto Correlation → Compare BTC/ETH correlations & moving averages", "completed": False},
            {"category": row['category'], "task": "Matrix Ops → Implement vectorized matrix multiplications", "completed": False},
            {"category": row['category'], "task": "Random Experiments → Simulate coin flips/dice & analyze", "completed": False}
        ])
    elif row['category'] == "Matplotlib":
        tasks_data.extend([
            {"category": row['category'], "task": "Stock Trends → Plot price trend with annotations & peaks", "completed": False},
            {"category": row['category'], "task": "Dashboard → Build multi-subplot analytics figure", "completed": False},
            {"category": row['category'], "task": "Style Guide → Create a custom plotting helper module", "completed": False}
        ])
    elif row['category'] == "Math Foundations":
        tasks_data.extend([
            {"category": row['category'], "task": "Vector Ops Notebook → Demonstrate dot/cross products & norms", "completed": False},
            {"category": row['category'], "task": "Stats Sampler → Simulate distributions & visualize histograms", "completed": False},
            {"category": row['category'], "task": "Gradient Demo → Numerical gradient on simple function", "completed": False}
        ])
    elif row['category'] == "OOP Foundations":
        tasks_data.extend([
            {"category": row['category'], "task": "Student Manager → Class-based CRUD with CSV persistence", "completed": False},
            {"category": row['category'], "task": "Shapes Library → Inheritance for Circle/Rect; compute area/perimeter", "completed": False},
            {"category": row['category'], "task": "Logger Wrapper → Context manager with __enter__/__exit__", "completed": False}
        ])
    elif row['category'] == "Portfolio Projects":
        tasks_data.extend([
            {"category": row['category'], "task": "Portfolio #1 → End-to-end sales/crypto analysis pipeline", "completed": False},
            {"category": row['category'], "task": "Portfolio #2 → Data Cleaning Toolkit with docs & tests", "completed": False},
            {"category": row['category'], "task": "Portfolio #3 → Two storytelling notebooks on Kaggle data", "completed": False}
        ])
    elif row['category'] == "scikit-learn":
        tasks_data.extend([
            {"category": row['category'], "task": "Titanic Classifier → Predict survival with CV & baseline", "completed": False},
            {"category": row['category'], "task": "House Prices → Feature engineering + regression models", "completed": False},
            {"category": row['category'], "task": "Iris Clustering → KMeans + elbow/silhouette analysis", "completed": False},
            {"category": row['category'], "task": "End-to-end Pipeline → ColumnTransformer + model + export", "completed": False}
        ])
    elif row['category'] == "Feature Engineering":
        tasks_data.extend([
            {"category": row['category'], "task": "Feature Cookoff → Compare encoders/scalers on 1 dataset", "completed": False},
            {"category": row['category'], "task": "Leakage Hunt → Find & fix leakage in a toy dataset", "completed": False},
            {"category": row['category'], "task": "Text TF-IDF → Build baseline text classifier", "completed": False}
        ])
    elif row['category'] == "Evaluation & Tuning":
        tasks_data.extend([
            {"category": row['category'], "task": "Metric Matrix → Benchmark models across metrics", "completed": False},
            {"category": row['category'], "task": "Hyperparam Sprint → GridSearchCV vs RandomizedSearchCV", "completed": False},
            {"category": row['category'], "task": "Explainability → Use SHAP on tabular model", "completed": False}
        ])
    elif row['category'] == "Project Engineering":
        tasks_data.extend([
            {"category": row['category'], "task": "Template Repo → Cookiecutter-style project skeleton", "completed": False},
            {"category": row['category'], "task": "Test Suite → Add pytest + coverage", "completed": False},
            {"category": row['category'], "task": "Simple API → Serve model via FastAPI", "completed": False}
        ])
    elif row['category'] == "Kaggle Track":
        tasks_data.extend([
            {"category": row['category'], "task": "Titanic CV → Build reliable CV; avoid overfit to LB", "completed": False},
            {"category": row['category'], "task": "House Prices FE → Feature engineering & stacking", "completed": False},
            {"category": row['category'], "task": "Report Notebook → Explain approach & errors", "completed": False}
        ])
    elif row['category'] == "PyTorch":
        tasks_data.extend([
            {"category": row['category'], "task": "MNIST From Scratch → Build full training loop + eval", "completed": False},
            {"category": row['category'], "task": "Augment & Overfit → Show effect of dropout/weight decay", "completed": False},
            {"category": row['category'], "task": "Checkpointing → Save/load to resume training", "completed": False}
        ])
    elif row['category'] == "CNNs":
        tasks_data.extend([
            {"category": row['category'], "task": "MNIST CNN → Train simple CNN from scratch", "completed": False},
            {"category": row['category'], "task": "CIFAR-10 Classifier → Augment + regularize + compare", "completed": False},
            {"category": row['category'], "task": "ResNet Transfer → Fine-tune on custom dataset", "completed": False},
            {"category": row['category'], "task": "MobileNet vs ResNet → Compare accuracy & latency", "completed": False}
        ])
    elif row['category'] == "RNNs & NLP":
        tasks_data.extend([
            {"category": row['category'], "task": "IMDB Sentiment → LSTM/GRU baseline with embeddings", "completed": False},
            {"category": row['category'], "task": "Attention Add-on → Add attention and compare scores", "completed": False},
            {"category": row['category'], "task": "Finetune Encoder → Use pretrained embeddings/encoders", "completed": False}
        ])
    elif row['category'] == "Transformers":
        tasks_data.extend([
            {"category": row['category'], "task": "Mini-Transformer → Train tiny model on toy dataset", "completed": False},
            {"category": row['category'], "task": "BERT Finetune → Downstream text classification", "completed": False},
            {"category": row['category'], "task": "Efficient Inference → Batching/padding + export", "completed": False}
        ])
    elif row['category'] == "GANs/SSL/RL":
        tasks_data.extend([
            {"category": row['category'], "task": "DCGAN Faces → Train DCGAN on small image set", "completed": False},
            {"category": row['category'], "task": "WGAN-GP Stability → Demonstrate improved training", "completed": False},
            {"category": row['category'], "task": "SimCLR Tiny → Contrastive pretraining on CIFAR-10", "completed": False}
        ])
    elif row['category'] == "MLOps":
        tasks_data.extend([
            {"category": row['category'], "task": "Serve Model → FastAPI endpoint with batching", "completed": False},
            {"category": row['category'], "task": "Dockerize → GPU-ready container to deploy API", "completed": False},
            {"category": row['category'], "task": "Track & Version → MLflow + DVC for one project", "completed": False}
        ])

# Create tasks DataFrame
tasks_df = pd.DataFrame(tasks_data)
tasks_df.to_csv(data_dir / "tasks.csv", index=False)

print("Data structure created successfully!")
print(f"Roadmap data saved to: {data_dir / 'roadmap_progress.csv'}")
print(f"Tasks data saved to: {data_dir / 'tasks.csv'}")