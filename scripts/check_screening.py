from graph.schemas import Paper
from slr.screening import screen_papers

papers = [
    Paper(
        paper_id="T1",
        title="Anomaly detection in chest X-ray images using convolutional autoencoders",
        abstract="We propose a convolutional autoencoder to detect anomalies in chest X-ray images. "
                 "The model is trained on healthy scans and flags abnormal lung regions on a public dataset.",
        year=2022,
    ),
    Paper(
        paper_id="T2",
        title="Forecasting electricity demand with transformer networks",
        abstract="We study short-term electricity load forecasting with transformer architectures. "
                 "Our model reduces the prediction error on three national datasets compared to LSTM baselines.",
        year=2022,
    ),
]

screen_papers(papers, 2018, delay=0)
for p in papers:
    print(p.paper_id, "| inclus:", p.included, "|", p.screening_reason)