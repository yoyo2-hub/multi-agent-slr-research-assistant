IDEA_PROFILE = {
    "problem": "Detecting anomalies in multivariate industrial sensor time series with very few labels",
    "objectives": ["Detect anomalies without labeled data", "Explain which sensors are responsible"],
    "methods": ["self-supervised contrastive learning", "graph attention over sensors"],
    "concepts": ["multivariate time series", "anomaly detection", "graph neural networks"],
    "research_questions": ["Does modeling sensor dependencies with a graph improve detection?"],
    "keywords": ["time series anomaly detection", "contrastive learning", "graph attention"],
    "expected_contributions": ["A graph-based contrastive detector", "An evaluation on industrial data"],
}

PAPER_PROFILES = [
    {
        "paper_id": "P1",
        "problem": "Unsupervised anomaly detection in multivariate time series",
        "method": "Contrastive learning with data augmentation",
        "contributions": ["A contrastive detector evaluated on several public datasets"],
        "limitations": ["Ignores explicit dependencies between sensors"],
        "evidence": [
            {"evidence_id": "E001", "paper_id": "P1", "section": "Method",
             "text": "We learn representations by contrasting augmented views of the same window."},
            {"evidence_id": "E002", "paper_id": "P1", "section": "Limitations",
             "text": "Our approach treats each channel independently and does not model inter-sensor relations."},
        ],
    },
    {
        "paper_id": "P2",
        "problem": "Benchmarking time series anomaly detection methods",
        "method": "Large-scale empirical comparison",
        "contributions": ["A unified benchmark of detection methods"],
        "limitations": ["Mostly public datasets, few industrial cases"],
        "evidence": [
            {"evidence_id": "E003", "paper_id": "P2", "section": "Limitations",
             "text": "The benchmark contains few real industrial deployments."},
        ],
    },
]