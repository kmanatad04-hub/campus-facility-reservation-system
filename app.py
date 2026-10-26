import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title='Campus Facility Reservation Monitoring', page_icon='🏫', layout='wide')
st.title('Campus Facility Reservation and Monitoring System')
st.caption('Decision support only. Facility administrators must review predictions before acting.')
REQUIRED = ['Reservation_ID','Facility_Type','User_Category','Reservation_Purpose','Lead_Time_Days','Participants','Facility_Capacity','Duration_Hours','Reservation_Weekday','Start_Hour','Usage_Status','Actual_Attendance','Cancellation_Days_Before']
PREDICTORS = ['Facility_Type','User_Category','Reservation_Purpose','Lead_Time_Days','Participants','Facility_Capacity','Duration_Hours','Reservation_Weekday','Start_Hour']
NUMERIC = ['Lead_Time_Days','Participants','Facility_Capacity','Duration_Hours','Reservation_Weekday','Start_Hour','Actual_Attendance','Cancellation_Days_Before']
CAT = ['Facility_Type','User_Category','Reservation_Purpose']

@st.cache_data
def clean_data(raw):
    df = raw.copy()
    for col in CAT + ['Usage_Status']:
        df[col] = df[col].apply(lambda v: v.strip().title() if isinstance(v,str) else v)
    df['Facility_Type'] = df['Facility_Type'].replace({'Lecturehall':'Lecture Hall','Meetingroom':'Meeting Room','Sportsfacility':'Sports Facility'})
    for col in NUMERIC: df[col] = pd.to_numeric(df[col], errors='coerce')
    checks = {'Lead_Time_Days': lambda x:x<0,'Participants':lambda x:x<=0,'Facility_Capacity':lambda x:x<=0,'Duration_Hours':lambda x:x<=0,'Reservation_Weekday':lambda x:~x.between(0,6),'Start_Hour':lambda x:~x.between(0,23),'Actual_Attendance':lambda x:x<0,'Cancellation_Days_Before':lambda x:x<0}
    for col,check in checks.items(): df.loc[df[col].notna() & check(df[col]),col] = np.nan
    df.loc[df['Actual_Attendance']>df['Participants'],'Actual_Attendance'] = np.nan
    df = df.drop_duplicates().copy()
    df['Lead_Time_Category'] = df['Lead_Time_Days'].apply(lambda x: np.nan if pd.isna(x) else 'Last Minute' if x<=2 else 'Short Lead' if x<=7 else 'Medium Lead' if x<=14 else 'Early Booking')
    df['Reservation_Outcome'] = df['Usage_Status'].map({'Used':'Completed','Cancelled':'Not Completed','No-Show':'Not Completed'})
    df['Attendance_Ratio'] = df['Actual_Attendance']/df['Participants']
    return df

@st.cache_resource
def fit_model(csv_bytes):
    raw = pd.read_csv(io.BytesIO(csv_bytes))
    df = clean_data(raw)
    data = df.dropna(subset=['Reservation_Outcome'])
    counts = data['Reservation_Outcome'].value_counts()
    if len(counts)<2 or counts.min()<2 or len(data)<10:
        raise ValueError('Training needs at least two outcome classes with at least two rows each and at least ten labeled rows.')
    X_train,X_test,y_train,y_test = train_test_split(data[PREDICTORS],data['Reservation_Outcome'],test_size=.20,random_state=42,stratify=data['Reservation_Outcome'])
    pre = ColumnTransformer([('numeric',Pipeline([('imputer',SimpleImputer(strategy='median'))]),[c for c in PREDICTORS if c not in CAT]),('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore'))]),CAT)])
    model = Pipeline([('preprocessor',pre),('classifier',DecisionTreeClassifier(max_depth=5,min_samples_leaf=5,class_weight='balanced',random_state=42))])
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    labels = ['Completed','Not Completed']
    metrics = {'Accuracy':accuracy_score(y_test,predictions),'Precision (Not Completed)':precision_score(y_test,predictions,pos_label='Not Completed',zero_division=0),'Recall (Not Completed)':recall_score(y_test,predictions,pos_label='Not Completed',zero_division=0),'F1 (Not Completed)':f1_score(y_test,predictions,pos_label='Not Completed',zero_division=0)}
    return df,model,metrics,confusion_matrix(y_test,predictions,labels=labels),labels,len(y_test)

uploaded = st.sidebar.file_uploader('Upload facility_reservations.csv',type=['csv'])
if uploaded is None:
    st.sidebar.caption('Using included sample dataset. Upload a CSV to replace it.')
page = st.sidebar.radio('Five modules',['1. Dashboard','2. Data Management','3. Data Analysis','4. Prediction','5. Model Evaluation'])
if uploaded is None:
    raw_bytes = (Path(__file__).parent / 'facility_reservations.csv').read_bytes()
else:
    raw_bytes = uploaded.getvalue()
try:
    original = pd.read_csv(io.BytesIO(raw_bytes))
    missing = [c for c in REQUIRED if c not in original.columns]
    if missing: st.error('Missing required columns: '+', '.join(missing)); st.stop()
    df = clean_data(original)
except Exception as exc:
    st.error(f'Unable to process CSV: {exc}'); st.stop()

if page.startswith('1.'):
    st.header('Dashboard — Current Dataset Status')
    a,b,c,d = st.columns(4)
    a.metric('Total reservations',len(df))
    b.metric('Completed',int((df.Reservation_Outcome=='Completed').sum()))
    c.metric('Cancelled / No-Show',int((df.Reservation_Outcome=='Not Completed').sum()))
    d.metric('Unknown outcome',int(df.Reservation_Outcome.isna().sum()))
    st.subheader('Observed reservation status')
    st.bar_chart(df['Usage_Status'].fillna('Unknown').value_counts())
    st.caption('Counts are observed historical records, not live facility availability.')
elif page.startswith('2.'):
    st.header('Data Management')
    st.write(f'Original rows: {len(original)} | Cleaned rows: {len(df)} | Exact duplicates removed: {len(original)-len(df)}')
    st.dataframe(df,use_container_width=True)
    st.subheader('Missing values after cleaning')
    st.dataframe(df.isna().sum().rename('Missing cells'))
    st.download_button('Download cleaned CSV',df.to_csv(index=False).encode('utf-8'),'facility_reservations_cleaned.csv','text/csv')
elif page.startswith('3.'):
    st.header('Data Analysis')
    col = st.selectbox('Group reservations by',['Facility_Type','User_Category','Reservation_Purpose','Lead_Time_Category'])
    st.bar_chart(df[col].fillna('Unknown').value_counts())
    summary = df.assign(Outcome=df['Reservation_Outcome'].fillna('Unknown')).groupby([col,'Outcome'],dropna=False).size().unstack(fill_value=0)
    st.subheader('Observed outcomes by group')
    st.dataframe(summary,use_container_width=True)
    st.bar_chart(summary)
else:
    try: clean,model,metrics,cm,labels,test_size = fit_model(raw_bytes)
    except Exception as exc: st.error(f'Model unavailable: {exc}'); st.stop()
    if page.startswith('4.'):
        st.header('Reservation Outcome Prediction')
        st.caption('Enter information known at reservation submission. No actual attendance or usage status is used.')
        with st.form('predict'):
            values = {}
            for col in CAT:
                choices = sorted(df[col].dropna().astype(str).unique().tolist())
                values[col] = st.selectbox(col.replace('_',' '),choices) if choices else st.text_input(col.replace('_',' '))
            values['Lead_Time_Days'] = st.number_input('Lead time (days)',min_value=0,value=5)
            values['Participants'] = st.number_input('Participants',min_value=1,value=25)
            values['Facility_Capacity'] = st.number_input('Facility capacity',min_value=1,value=50)
            values['Duration_Hours'] = st.number_input('Duration (hours)',min_value=0.1,value=2.0)
            values['Reservation_Weekday'] = st.number_input('Reservation weekday (0–6, as encoded in CSV)',min_value=0,max_value=6,value=0)
            values['Start_Hour'] = st.number_input('Start hour (0–23)',min_value=0,max_value=23,value=9)
            submitted = st.form_submit_button('Predict reservation outcome')
        if submitted:
            sample = pd.DataFrame([values],columns=PREDICTORS)
            label = model.predict(sample)[0]
            probabilities = dict(zip(model.classes_,model.predict_proba(sample)[0]))
            st.success(f'Predicted outcome: {label}')
            st.write(f'Estimated Not Completed probability: {probabilities.get("Not Completed",0):.1%}')
            st.caption('This probability is a model estimate, not a verified real-world likelihood. Human review is required.')
        st.subheader('Five sample predictions from uploaded records')
        examples = df[PREDICTORS].head(5).copy()
        if len(examples):
            examples.insert(0,'Reservation_ID',df['Reservation_ID'].head(5).values)
            examples['Predicted_Outcome'] = model.predict(df[PREDICTORS].head(5))
            examples['Estimated_Not_Completed_Probability'] = model.predict_proba(df[PREDICTORS].head(5))[:,list(model.classes_).index('Not Completed')].round(3)
            st.dataframe(examples,use_container_width=True)
            st.download_button('Download five predictions',examples.to_csv(index=False).encode(),'five_sample_predictions.csv','text/csv')
    else:
        st.header('Model Evaluation')
        st.write(f'Held-out test records: {test_size} (80/20 stratified split; random_state=42)')
        for label,value in metrics.items(): st.metric(label,f'{value:.3f}')
        fig,ax=plt.subplots(figsize=(5,4))
        ConfusionMatrixDisplay(cm,display_labels=labels).plot(ax=ax,cmap='Greys',colorbar=False)
        st.pyplot(fig)
        st.caption('Test metrics depend on the uploaded dataset; they are not predetermined.')
