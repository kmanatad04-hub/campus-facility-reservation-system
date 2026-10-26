# Campus Facility Reservation and Monitoring System

Streamlit conversion of the original midterm Jupyter notebook. Upload a CSV with the original notebook's 13 required fields. Five modules: Dashboard, Data Management, Data Analysis, Prediction, Model Evaluation.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, `README.md`, and `original_midterm.ipynb`.
2. At https://share.streamlit.io/ choose **Create app**, connect your GitHub repository, set the entrypoint to `app.py`, and deploy.
3. Copy your actual repository and deployment URLs into `SUBMISSION.md`.
4. Upload the same real dataset in the deployed app. Capture screenshots of all five modules and five sample predictions.

No real dataset was included with the notebook upload, so sample predictions and deployment screenshots must be generated after you upload your CSV.

Included `facility_reservations.csv` is converted from the supplied Excel workbook and loads automatically. Uploading a CSV in the sidebar overrides it. `five_sample_predictions.csv` contains five model-generated examples.
