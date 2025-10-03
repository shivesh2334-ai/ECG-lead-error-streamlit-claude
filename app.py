import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps
import io
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Check if OpenCV is available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    st.warning("OpenCV (cv2) not available. Using PIL for basic image processing.")

# Set page config
st.set_page_config(
    page_title="ECG Lead Misplacement Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class ECGLeadMisplacementDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        self.misplacement_types = [
            'Correct',
            'RA-LA Reversal',
            'RA-LL Reversal', 
            'LA-LL Reversal',
            'Limb-Neutral Reversal',
            'Precordial Misplacement'
        ]
        
    def extract_features_from_ecg(self, ecg_data):
        features = {}
        features['lead_I_negative_p'] = 1 if np.mean(ecg_data.get('I', [0])) < 0 else 0
        features['lead_I_negative_qrs'] = 1 if np.min(ecg_data.get('I', [0])) < -0.1 else 0
        features['lead_II_negative_p'] = 1 if np.mean(ecg_data.get('II', [0])) < 0 else 0
        features['lead_II_negative_qrs'] = 1 if np.min(ecg_data.get('II', [0])) < -0.1 else 0
        features['lead_II_flat'] = 1 if np.std(ecg_data.get('II', [0])) < 0.05 else 0
        features['lead_III_flat'] = 1 if np.std(ecg_data.get('III', [0])) < 0.05 else 0
        features['avr_positive_p'] = 1 if np.mean(ecg_data.get('aVR', [0])) > 0 else 0
        features['avr_positive_qrs'] = 1 if np.max(ecg_data.get('aVR', [0])) > 0.1 else 0
        p_lead_I = np.mean(ecg_data.get('I', [0])[:int(len(ecg_data.get('I', [0]))*0.3)])
        p_lead_II = np.mean(ecg_data.get('II', [0])[:int(len(ecg_data.get('II', [0]))*0.3)])
        p_lead_III = np.mean(ecg_data.get('III', [0])[:int(len(ecg_data.get('III', [0]))*0.3)])
        features['p_axis_abnormal'] = 1 if p_lead_I > p_lead_II else 0
        features['p_wave_terminal_positive_III'] = 1 if p_lead_III > 0.05 else 0
        qrs_lead_I = np.max(ecg_data.get('I', [0])) - np.min(ecg_data.get('I', [0]))
        qrs_lead_II = np.max(ecg_data.get('II', [0])) - np.min(ecg_data.get('II', [0]))
        features['qrs_axis_shift'] = abs(qrs_lead_I - qrs_lead_II) / max(qrs_lead_I, qrs_lead_II, 0.1)
        precordial_leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        r_wave_progression = [np.max(ecg_data.get(lead, [0])) for lead in precordial_leads]
        normal_progression = all(r_wave_progression[i] <= r_wave_progression[i+1] for i in range(3))
        features['abnormal_r_progression'] = 0 if normal_progression else 1
        correlations = []
        for i, lead1 in enumerate(self.lead_names):
            for j, lead2 in enumerate(self.lead_names):
                if i < j and lead1 in ecg_data and lead2 in ecg_data:
                    if len(ecg_data[lead1]) > 1 and len(ecg_data[lead2]) > 1:
                        corr, _ = pearsonr(ecg_data[lead1], ecg_data[lead2])
                        correlations.append(abs(corr))
        features['mean_correlation'] = np.mean(correlations) if correlations else 0
        features['min_correlation'] = np.min(correlations) if correlations else 0
        return features
    
    def detect_specific_misplacement(self, features):
        if (features['lead_I_negative_p'] and features['lead_I_negative_qrs'] and features['avr_positive_p']):
            return 'RA-LA Reversal', 0.95
        if features['lead_II_negative_p'] and features['lead_II_negative_qrs']:
            return 'RA-LL Reversal', 0.90
        if (features['p_axis_abnormal'] or features['p_wave_terminal_positive_III']):
            return 'LA-LL Reversal', 0.75
        if (features['lead_II_flat'] or features['lead_III_flat']):
            return 'Limb-Neutral Reversal', 0.85
        if features['abnormal_r_progression']:
            return 'Precordial Misplacement', 0.70
        if (features['qrs_axis_shift'] > 0.3 or features['mean_correlation'] < 0.6):
            return 'Possible Misplacement', 0.60
        return 'Correct', 0.95
    
    def train_model(self, training_data):
        features_list = []
        labels = []
        for ecg_record in training_data:
            features = self.extract_features_from_ecg(ecg_record['data'])
            features_list.append(list(features.values()))
            labels.append(ecg_record['label'])
        X = np.array(features_list)
        y = np.array(labels)
        X_scaled = self.scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy, classification_report(y_test, y_pred)

def generate_synthetic_ecg_data():
    np.random.seed(42)
    t = np.linspace(0, 1, 1000)
    p_wave = 0.1 * np.exp(-((t - 0.15) / 0.05) ** 2)
    qrs_complex = 0.8 * np.exp(-((t - 0.4) / 0.02) ** 2) - 0.2 * np.exp(-((t - 0.38) / 0.01) ** 2)
    t_wave = 0.15 * np.exp(-((t - 0.7) / 0.08) ** 2)
    base_ecg = p_wave + qrs_complex + t_wave + 0.02 * np.random.normal(0, 1, len(t))
    training_data = []
    correct_ecg = {
        'I': base_ecg + 0.1 * np.random.normal(0, 1, len(t)),
        'II': base_ecg * 1.2 + 0.1 * np.random.normal(0, 1, len(t)),
        'III': base_ecg * 0.8 + 0.1 * np.random.normal(0, 1, len(t)),
        'aVR': -base_ecg * 0.5 + 0.05 * np.random.normal(0, 1, len(t)),
        'aVL': base_ecg * 0.6 + 0.05 * np.random.normal(0, 1, len(t)),
        'aVF': base_ecg * 0.9 + 0.05 * np.random.normal(0, 1, len(t)),
        'V1': base_ecg * 0.4 + 0.05 * np.random.normal(0, 1, len(t)),
        'V2': base_ecg * 0.6 + 0.05 * np.random.normal(0, 1, len(t)),
        'V3': base_ecg * 0.8 + 0.05 * np.random.normal(0, 1, len(t)),
        'V4': base_ecg * 1.0 + 0.05 * np.random.normal(0, 1, len(t)),
        'V5': base_ecg * 1.1 + 0.05 * np.random.normal(0, 1, len(t)),
        'V6': base_ecg * 1.0 + 0.05 * np.random.normal(0, 1, len(t))
    }
    for i in range(50):
        training_data.append({'data': correct_ecg, 'label': 'Correct'})
    ra_la_reversal = correct_ecg.copy()
    ra_la_reversal['I'] = -correct_ecg['I']
    ra_la_reversal['aVR'] = -ra_la_reversal['aVR']
    for i in range(30):
        training_data.append({'data': ra_la_reversal, 'label': 'RA-LA Reversal'})
    ra_ll_reversal = correct_ecg.copy()
    ra_ll_reversal['II'] = -correct_ecg['II']
    for i in range(20):
        training_data.append({'data': ra_ll_reversal, 'label': 'RA-LL Reversal'})
    la_ll_reversal = correct_ecg.copy()
    la_ll_reversal['I'] = la_ll_reversal['II'] * 0.8
    la_ll_reversal['aVL'], la_ll_reversal['aVF'] = la_ll_reversal['aVF'], la_ll_reversal['aVL']
    for i in range(20):
        training_data.append({'data': la_ll_reversal, 'label': 'LA-LL Reversal'})
    limb_neutral_reversal = correct_ecg.copy()
    limb_neutral_reversal['II'] = np.zeros_like(correct_ecg['II']) + 0.01 * np.random.normal(0, 1, len(t))
    for i in range(15):
        training_data.append({'data': limb_neutral_reversal, 'label': 'Limb-Neutral Reversal'})
    precordial_misplacement = correct_ecg.copy()
    precordial_misplacement['V2'], precordial_misplacement['V4'] = precordial_misplacement['V4'], precordial_misplacement['V2']
    for i in range(15):
        training_data.append({'data': precordial_misplacement, 'label': 'Precordial Misplacement'})
    return training_data

def process_uploaded_file(uploaded_file):
    try:
        file_type = uploaded_file.type
        if file_type in ['image/jpeg', 'image/jpg', 'image/png']:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded ECG Image", use_column_width=True)
            img_array = np.array(image)
            if CV2_AVAILABLE and len(img_array.shape) == 3:
                gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                signals = extract_signals_from_image_cv2(gray_image)
            else:
                if len(img_array.shape) == 3:
                    gray_image = ImageOps.grayscale(image)
                    gray_array = np.array(gray_image)
                else:
                    gray_array = img_array
                signals = extract_signals_from_image_pil(gray_array)
            return signals
        elif file_type == 'application/pdf':
            st.warning("PDF processing requires additional libraries. Please convert to image format or use a PDF to image converter.")
            return None
        else:
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            try:
                data = pd.read_csv(io.StringIO(content))
                return convert_dataframe_to_ecg(data)
            except Exception as e:
                st.error(f"Could not parse file format: {str(e)}. Please upload ECG data in CSV format or as an image.")
                return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def extract_signals_from_image_pil(image_array):
    height, width = image_array.shape
    leads = {}
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    rows, cols = 3, 4
    for i, lead_name in enumerate(lead_names):
        row = i // cols
        col = i % cols
        start_row = int(row * height / rows)
        end_row = int((row + 1) * height / rows)
        start_col = int(col * width / cols)
        end_col = int((col + 1) * width / cols)
        region = image_array[start_row:end_row, start_col:end_col]
        signal_data = []
        for column in range(min(region.shape[1], 500)):
            if region.shape[1] > column:
                col_data = region[:, column]
                if len(col_data) > 0:
                    min_row = np.argmin(col_data)
                    normalized_value = (region.shape[0] - min_row) / region.shape[0] - 0.5
                    signal_data.append(normalized_value)
                else:
                    signal_data.append(0)
            else:
                signal_data.append(0)
        while len(signal_data) < 100:
            signal_data.extend(signal_data)
        leads[lead_name] = signal_data[:1000]
    return leads

def extract_signals_from_image_cv2(image):
    height, width = image.shape
    leads = {}
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    rows, cols = 3, 4
    for i, lead_name in enumerate(lead_names):
        row = i // cols
        col = i % cols
        start_row = int(row * height / rows)
        end_row = int((row + 1) * height / rows)
        start_col = int(col * width / cols)
        end_col = int((col + 1) * width / cols)
        region = image[start_row:end_row, start_col:end_col]
        signal = []
        for column in range(region.shape[1]):
            col_data = region[:, column]
            min_row = np.argmin(col_data)
            signal.append((region.shape[0] - min_row) / region.shape[0] - 0.5)
        leads[lead_name] = signal
    return leads

def convert_dataframe_to_ecg(df):
    ecg_data = {}
    lead_mapping = {
        'I': ['I', 'Lead_I', 'lead_1', 'LEAD_I'],
        'II': ['II', 'Lead_II', 'lead_2', 'LEAD_II'],
        'III': ['III', 'Lead_III', 'lead_3', 'LEAD_III'],
        'aVR': ['aVR', 'AVR', 'Lead_aVR', 'avr', 'LEAD_AVR'],
        'aVL': ['aVL', 'AVL', 'Lead_aVL', 'avl', 'LEAD_AVL'],
        'aVF': ['aVF', 'AVF', 'Lead_aVF', 'avf', 'LEAD_AVF'],
        'V1': ['V1', 'Lead_V1', 'v1', 'LEAD_V1'],
        'V2': ['V2', 'Lead_V2', 'v2', 'LEAD_V2'],
        'V3': ['V3', 'Lead_V3', 'v3', 'LEAD_V3'],
        'V4': ['V4', 'Lead_V4', 'v4', 'LEAD_V4'],
        'V5': ['V5', 'Lead_V5', 'v5', 'LEAD_V5'],
        'V6': ['V6', 'Lead_V6', 'v6', 'LEAD_V6']
    }
    for standard_name, possible_names in lead_mapping.items():
        for col_name in possible_names:
            if col_name in df.columns:
                ecg_data[standard_name] = df[col_name].values.tolist()
                break
    return ecg_data

def show_requirements():
    requirements_content = """
streamlit==1.28.1
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
Pillow==10.0.0
scipy==1.11.1
opencv-python-headless==4.8.0.76
"""
    return requirements_content

def main():
    st.markdown('<h1 class="main-header">🫀 ECG Lead Misplacement Detection System</h1>', unsafe_allow_html=True)
    with st.expander("📋 Deployment Requirements"):
        st.markdown("### For Streamlit Cloud Deployment")
        st.markdown("Create a `requirements.txt` file with the following content:")
        st.code(show_requirements(), language="text")
        st.markdown("### Alternative Installation")
        st.markdown("If OpenCV issues persist, you can use the app without advanced image processing features.")
    st.markdown("""
    <div class="info-box">
    <h3>About This Application</h3>
    <p>This application detects incorrect electrode cable connections during ECG recording based on the research by 
    Batchvarov et al. (2007). It implements automated detection algorithms using machine learning and rule-based 
    approaches to identify various types of lead misplacements.</p>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.header("🔧 Configuration")
    detection_method = st.sidebar.selectbox(
        "Select Detection Method",
        ["Rule-based Detection", "Machine Learning", "Both Methods"]
    )
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Minimum confidence level for misplacement detection"
    )
    if 'detector' not in st.session_state:
        st.session_state.detector = ECGLeadMisplacementDetector()
        with st.spinner("Initializing ML model with synthetic training data..."):
            training_data = generate_synthetic_ecg_data()
            accuracy, report = st.session_state.detector.train_model(training_data)
            st.session_state.model_accuracy = accuracy
            st.session_state.model_report = report
    st.sidebar.markdown("### 📊 Model Performance")
    st.sidebar.metric("Accuracy", f"{st.session_state.model_accuracy:.2%}")
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Analyze", "📚 Misplacement Types", "🔍 Detection Rules", "📈 Model Details"])
    # ... [rest of main() as in your working code, but avoid duplication and overlap]
    # For brevity, keep only one version of each tab, function, and class.

if __name__ == "__main__":
    main()
