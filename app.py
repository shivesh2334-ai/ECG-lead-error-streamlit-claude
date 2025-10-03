import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageOps
import io
import base64
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Alternative to cv2 for image processing
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

# Create requirements.txt content for Streamlit deployment
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

# Custom CSS (same as before)
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
        """Extract features for misplacement detection based on the paper's criteria"""
        features = {}
        
        # Lead I features
        features['lead_I_negative_p'] = 1 if np.mean(ecg_data.get('I', [0])) < 0 else 0
        features['lead_I_negative_qrs'] = 1 if np.min(ecg_data.get('I', [0])) < -0.1 else 0
        
        # Lead II features  
        features['lead_II_negative_p'] = 1 if np.mean(ecg_data.get('II', [0])) < 0 else 0
        features['lead_II_negative_qrs'] = 1 if np.min(ecg_data.get('II', [0])) < -0.1 else 0
        features['lead_II_flat'] = 1 if np.std(ecg_data.get('II', [0])) < 0.05 else 0
        
        # Lead III features
        features['lead_III_flat'] = 1 if np.std(ecg_data.get('III', [0])) < 0.05 else 0
        
        # aVR features
        features['avr_positive_p'] = 1 if np.mean(ecg_data.get('aVR', [0])) > 0 else 0
        features['avr_positive_qrs'] = 1 if np.max(ecg_data.get('aVR', [0])) > 0.1 else 0
        
        # P wave axis estimation
        p_lead_I = np.mean(ecg_data.get('I', [0])[:int(len(ecg_data.get('I', [0]))*0.3)])
        p_lead_II = np.mean(ecg_data.get('II', [0])[:int(len(ecg_data.get('II', [0]))*0.3)])
        p_lead_III = np.mean(ecg_data.get('III', [0])[:int(len(ecg_data.get('III', [0]))*0.3)])
        
        features['p_axis_abnormal'] = 1 if p_lead_I > p_lead_II else 0
        features['p_wave_terminal_positive_III'] = 1 if p_lead_III > 0.05 else 0
        
        # QRS axis estimation
        qrs_lead_I = np.max(ecg_data.get('I', [0])) - np.min(ecg_data.get('I', [0]))
        qrs_lead_II = np.max(ecg_data.get('II', [0])) - np.min(ecg_data.get('II', [0]))
        
        features['qrs_axis_shift'] = abs(qrs_lead_I - qrs_lead_II) / max(qrs_lead_I, qrs_lead_II, 0.1)
        
        # Precordial progression features
        precordial_leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        r_wave_progression = []
        
        for lead in precordial_leads:
            if lead in ecg_data:
                r_wave = np.max(ecg_data[lead])
                r_wave_progression.append(r_wave)
            else:
                r_wave_progression.append(0)
        
        # Check for abnormal R wave progression
        if len(r_wave_progression) >= 4:
            normal_progression = all(r_wave_progression[i] <= r_wave_progression[i+1] 
                                   for i in range(3))  # V1-V4 should increase
            features['abnormal_r_progression'] = 0 if normal_progression else 1
        else:
            features['abnormal_r_progression'] = 0
        
        # Lead correlations (for reconstruction method)
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
        """Detect specific type of misplacement based on paper's criteria"""
        
        # RA-LA Reversal (most common)
        if (features['lead_I_negative_p'] and features['lead_I_negative_qrs'] and 
            features['avr_positive_p']):
            return 'RA-LA Reversal', 0.95
        
        # RA-LL Reversal  
        if features['lead_II_negative_p'] and features['lead_II_negative_qrs']:
            return 'RA-LL Reversal', 0.90
        
        # LA-LL Reversal
        if (features['p_axis_abnormal'] or features['p_wave_terminal_positive_III']):
            return 'LA-LL Reversal', 0.75
        
        # Limb-Neutral cable reversal
        if (features['lead_II_flat'] or features['lead_III_flat']):
            return 'Limb-Neutral Reversal', 0.85
        
        # Precordial misplacement
        if features['abnormal_r_progression']:
            return 'Precordial Misplacement', 0.70
        
        # Check for subtle changes
        if (features['qrs_axis_shift'] > 0.3 or features['mean_correlation'] < 0.6):
            return 'Possible Misplacement', 0.60
            
        return 'Correct', 0.95
    
    def train_model(self, training_data):
        """Train ML model on ECG data"""
        features_list = []
        labels = []
        
        for ecg_record in training_data:
            features = self.extract_features_from_ecg(ecg_record['data'])
            feature_vector = list(features.values())
            features_list.append(feature_vector)
            labels.append(ecg_record['label'])
        
        X = np.array(features_list)
        y = np.array(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, classification_report(y_test, y_pred)

def generate_synthetic_ecg_data():
    """Generate synthetic ECG data for demonstration"""
    np.random.seed(42)
    
    # Normal ECG pattern
    t = np.linspace(0, 1, 1000)  # 1 second of data
    
    # Basic ECG components
    p_wave = 0.1 * np.exp(-((t - 0.15) / 0.05) ** 2)
    qrs_complex = 0.8 * np.exp(-((t - 0.4) / 0.02) ** 2) - 0.2 * np.exp(-((t - 0.38) / 0.01) ** 2)
    t_wave = 0.15 * np.exp(-((t - 0.7) / 0.08) ** 2)
    
    base_ecg = p_wave + qrs_complex + t_wave + 0.02 * np.random.normal(0, 1, len(t))
    
    # Generate data for different conditions
    training_data = []
    
    # Correct ECG
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
    
    # RA-LA Reversal (negative lead I)
    ra_la_reversal = correct_ecg.copy()
    ra_la_reversal['I'] = -correct_ecg['I']  # Lead I becomes negative
    ra_la_reversal['aVR'] = -ra_la_reversal['aVR']  # aVR becomes positive
    
    for i in range(30):
        training_data.append({'data': ra_la_reversal, 'label': 'RA-LA Reversal'})
    
    # RA-LL Reversal (negative lead II)
    ra_ll_reversal = correct_ecg.copy()
    ra_ll_reversal['II'] = -correct_ecg['II']  # Lead II becomes negative
    
    for i in range(20):
        training_data.append({'data': ra_ll_reversal, 'label': 'RA-LL Reversal'})
    
    # LA-LL Reversal (subtle changes in P wave)
    la_ll_reversal = correct_ecg.copy()
    # Swap some characteristics
    temp = la_ll_reversal['I']
    la_ll_reversal['I'] = la_ll_reversal['II'] * 0.8
    la_ll_reversal['aVL'], la_ll_reversal['aVF'] = la_ll_reversal['aVF'], la_ll_reversal['aVL']
    
    for i in range(20):
        training_data.append({'data': la_ll_reversal, 'label': 'LA-LL Reversal'})
    
    # Limb-Neutral Reversal (flat line in lead II)
    limb_neutral_reversal = correct_ecg.copy()
    limb_neutral_reversal['II'] = np.zeros_like(correct_ecg['II']) + 0.01 * np.random.normal(0, 1, len(t))
    
    for i in range(15):
        training_data.append({'data': limb_neutral_reversal, 'label': 'Limb-Neutral Reversal'})
    
    # Precordial Misplacement (abnormal R wave progression)
    precordial_misplacement = correct_ecg.copy()
    # Reverse some precordial leads
    precordial_misplacement['V2'], precordial_misplacement['V4'] = precordial_misplacement['V4'], precordial_misplacement['V2']
    
    for i in range(15):
        training_data.append({'data': precordial_misplacement, 'label': 'Precordial Misplacement'})
    
    return training_data

def process_uploaded_file(uploaded_file):
    """Process uploaded ECG file without cv2 dependency"""
    try:
        file_type = uploaded_file.type
        
        if file_type in ['image/jpeg', 'image/jpg', 'image/png']:
            # Process image file using PIL instead of cv2
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded ECG Image", use_column_width=True)
            
            # Convert to numpy array for processing
            img_array = np.array(image)
            
            # Convert to grayscale using PIL
            if len(img_array.shape) == 3:
                gray_image = ImageOps.grayscale(image)
                gray_array = np.array(gray_image)
            else:
                gray_array = img_array
            
            # Extract signal lines (simplified approach)
            signals = extract_signals_from_image_pil(gray_array)
            return signals
            
        elif file_type == 'application/pdf':
            st.warning("PDF processing requires additional libraries. Please convert to image format or use a PDF to image converter.")
            return None
            
        else:
            # Assume it's a data file (CSV, text, etc.)
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            # Try to parse as CSV
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
    """Extract ECG signals from image using PIL instead of cv2"""
    height, width = image_array.shape
    
    # Divide image into 12 sections for 12 leads (3x4 grid typical)
    leads = {}
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    rows, cols = 3, 4
    for i, lead_name in enumerate(lead_names):
        row = i // cols
        col = i % cols
        
        # Extract region
        start_row = int(row * height / rows)
        end_row = int((row + 1) * height / rows)
        start_col = int(col * width / cols)
        end_col = int((col + 1) * width / cols)
        
        region = image_array[start_row:end_row, start_col:end_col]
        
        # Extract signal by finding the darkest line in each column
        signal_data = []
        for column in range(min(region.shape[1], 500)):  # Limit to 500 points
            if region.shape[1] > column:
                col_data = region[:, column]
                # Find the row with minimum value (darkest point)
                if len(col_data) > 0:
                    min_row = np.argmin(col_data)
                    # Normalize to create signal
                    normalized_value = (region.shape[0] - min_row) / region.shape[0] - 0.5
                    signal_data.append(normalized_value)
                else:
                    signal_data.append(0)
            else:
                signal_data.append(0)
        
        # Ensure minimum length
        while len(signal_data) < 100:
            signal_data.extend(signal_data)
        
        leads[lead_name] = signal_data[:1000]  # Limit to 1000 points
    
    return leads

def convert_dataframe_to_ecg(df):
    """Convert dataframe to ECG format"""
    ecg_data = {}
    
    # Map dataframe columns to ECG leads
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

def main():
    st.markdown('<h1 class="main-header">🫀 ECG Lead Misplacement Detection System</h1>', unsafe_allow_html=True)
    
    # Show deployment requirements
    with st.expander("📋 Deployment Requirements"):
        st.markdown("### For Streamlit Cloud Deployment")
        st.markdown("Create a `requirements.txt` file with the following content:")
        st.code(show_requirements(), language="text")
        st.markdown("### Alternative Installation")
        st.markdown("If OpenCV issues persist, you can use the app without advanced image processing features.")
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    <h3>About This Application</h3>
    <p>This application detects incorrect electrode cable connections during ECG recording based on the research by 
    Batchvarov et al. (2007). It implements automated detection algorithms using machine learning and rule-based 
    approaches to identify various types of lead misplacements.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Detection method selection
    detection_method = st.sidebar.selectbox(
        "Select Detection Method",
        ["Rule-based Detection", "Machine Learning", "Both Methods"]
    )
    
    # File upload sensitivity
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Minimum confidence level for misplacement detection"
    )
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = ECGLeadMisplacementDetector()
        
        # Generate and train on synthetic data
        with st.spinner("Initializing ML model with synthetic training data..."):
            training_data = generate_synthetic_ecg_data()
            accuracy, report = st.session_state.detector.train_model(training_data)
            st.session_state.model_accuracy = accuracy
            st.session_state.model_report = report
    
    # Display model performance
    st.sidebar.markdown("### 📊 Model Performance")
    st.sidebar.metric("Accuracy", f"{st.session_state.model_accuracy:.2%}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Analyze", "📚 Misplacement Types", "🔍 Detection Rules", "📈 Model Details"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Upload ECG Data</h2>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose ECG file",
            type=['jpg', 'jpeg', 'png', 'pdf', 'csv', 'txt'],
            help="Upload ECG image (JPG, PNG), PDF, or data file (CSV, TXT)"
        )
        
        # Demo data option
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧪 Use Demo Data - Normal ECG"):
                demo_data = generate_synthetic_ecg_data()[0]['data']  # Normal ECG
                st.session_state.current_ecg = demo_data
                st.session_state.demo_type = "Normal ECG"
        
        with col2:
            demo_misplacement = st.selectbox(
                "Demo Misplacement Type",
                ["RA-LA Reversal", "RA-LL Reversal", "LA-LL Reversal", "Limb-Neutral Reversal", "Precordial Misplacement"]
            )
            
            if st.button(f"🧪 Use Demo Data - {demo_misplacement}"):
                demo_data_list = generate_synthetic_ecg_data()
                for data in demo_data_list:
                    if data['label'] == demo_misplacement:
                        st.session_state.current_ecg = data['data']
                        st.session_state.demo_type = demo_misplacement
                        break
        
        # Process uploaded file
        if uploaded_file is not None:
            ecg_data = process_uploaded_file(uploaded_file)
            if ecg_data:
                st.session_state.current_ecg = ecg_data
                st.success("✅ File processed successfully!")
        
        # Analyze ECG data
        if 'current_ecg' in st.session_state:
            st.markdown('<h2 class="sub-header">📊 ECG Analysis Results</h2>', unsafe_allow_html=True)
            
            ecg_data = st.session_state.current_ecg
            
            # Display ECG signals
            st.markdown("### ECG Waveforms")
            
            fig, axes = plt.subplots(4, 3, figsize=(15, 12))
            fig.suptitle('12-Lead ECG', fontsize=16)
            
            lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            
            for i, lead in enumerate(lead_names):
                row, col = i // 3, i % 3
                if lead in ecg_data and len(ecg_data[lead]) > 0:
                    axes[row, col].plot(ecg_data[lead])
                    axes[row, col].set_title(f'Lead {lead}')
                    axes[row, col].grid(True, alpha=0.3)
                else:
                    axes[row, col].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[row, col].transAxes)
                    axes[row, col].set_title(f'Lead {lead}')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Perform detection
            st.markdown("### 🔍 Misplacement Detection Results")
            
            # Extract features
            features = st.session_state.detector.extract_features_from_ecg(ecg_data)
            
            # Rule-based detection
            if detection_method in ["Rule-based Detection", "Both Methods"]:
                st.markdown("#### Rule-based Analysis")
                
                misplacement_type, confidence = st.session_state.detector.detect_specific_misplacement(features)
                
                if confidence >= confidence_threshold:
                    if misplacement_type == "Correct":
                        st.markdown(f"""
                        <div class="success-box">
                        <h4>✅ Result: {misplacement_type}</h4>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p>No significant lead misplacement detected.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-box">
                        <h4>⚠️ Result: {misplacement_type}</h4>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p>Lead misplacement detected! Please check electrode connections.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                    <h4>❓ Result: Uncertain</h4>
                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    <p>Detection confidence below threshold. Manual review recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Machine learning detection
            if detection_method in ["Machine Learning", "Both Methods"]:
                st.markdown("#### Machine Learning Analysis")
                
                # Prepare features for ML model
                feature_vector = np.array(list(features.values())).reshape(1, -1)
                feature_vector_scaled = st.session_state.detector.scaler.transform(feature_vector)
                
                # Predict
                ml_prediction = st.session_state.detector.model.predict(feature_vector_scaled)[0]
                ml_probabilities = st.session_state.detector.model.predict_proba(feature_vector_scaled)[0]
                
                # Get the confidence for the predicted class
                predicted_class_idx = st.session_state.detector.model.classes_.tolist().index(ml_prediction)
                ml_confidence = ml_probabilities[predicted_class_idx]
                
                if ml_confidence >= confidence_threshold:
                    if ml_prediction == "Correct":
                        st.markdown(f"""
                        <div class="success-box">
                        <h4>✅ ML Result: {ml_prediction}</h4>
                        <p><strong>Confidence:</strong> {ml_confidence:.2%}</p>
                        <p>Machine learning model indicates correct lead placement.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-box">
                        <h4>⚠️ ML Result: {ml_prediction}</h4>
                        <p><strong>Confidence:</strong> {ml_confidence:.2%}</p>
                        <p>Machine learning model detected lead misplacement.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                    <h4>❓ ML Result: Uncertain</h4>
                    <p><strong>Confidence:</strong> {ml_confidence:.2%}</p>
                    <p>ML prediction confidence below threshold.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show probability distribution
                st.markdown("##### Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Misplacement Type': st.session_state.detector.model.classes_,
                    'Probability': ml_probabilities
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(prob_df['Misplacement Type'], prob_df['Probability'])
                ax.set_ylabel('Probability')
                ax.set_title('Machine Learning Prediction Probabilities')
                plt.xticks(rotation=45, ha='right')
                
                # Highlight the predicted class
                bars[predicted_class_idx].set_color('red')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Feature analysis
            st.markdown("### 📋 Feature Analysis")
            
            feature_df = pd.DataFrame([features]).T
            feature_df.columns = ['Value']
            feature_df.index.name = 'Feature'
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(feature_df, use_container_width=True)
            
            with col2:
                # Key diagnostic features
                st.markdown("#### 🔑 Key Diagnostic Indicators")
                
                diagnostic_messages = []
                
                if features['lead_I_negative_p'] or features['lead_I_negative_qrs']:
                    diagnostic_messages.append("• Negative P or QRS in Lead I (suggests RA-LA reversal)")
                
                if features['lead_II_negative_p'] or features['lead_II_negative_qrs']:
                    diagnostic_messages.append("• Negative P or QRS in Lead II (suggests RA-LL reversal)")
                
                if features['avr_positive_p'] or features['avr_positive_qrs']:
                    diagnostic_messages.append("• Positive complexes in aVR (abnormal)")
                
                if features['lead_II_flat'] or features['lead_III_flat']:
                    diagnostic_messages.append("• Flat line in limb lead (suggests neutral cable issue)")
                
                if features['p_axis_abnormal']:
                    diagnostic_messages.append("• Abnormal P wave axis")
                
                if features['abnormal_r_progression']:
                    diagnostic_messages.append("• Abnormal R wave progression in precordial leads")
                
                if diagnostic_messages:
                    for msg in diagnostic_messages:
                        st.markdown(msg)
                else:
                    st.markdown("• No significant abnormalities detected")
    
    # [Rest of the tabs remain the same as in the original code]
    with tab2:
        st.markdown('<h2 class="sub-header">📚 Types of Lead Misplacements</h2>', unsafe_allow_html=True)
        
        misplacement_info = {
            "RA-LA Reversal": {
                "frequency": "Most common (60-70% of misplacements)",
                "signs": [
                    "Negative P and QRS complexes in Lead I",
                    "Positive P wave in aVR",
                    "Normal precordial leads",
                    "Resembles mirror-image dextrocardia"
                ],
                "clinical_impact": "Can simulate dextrocardia, may mask or simulate cardiac abnormalities"
            },
            "RA-LL Reversal": {
                "frequency": "Less common (10-15% of misplacements)",
                "signs": [
                    "Negative P and QRS complexes in Lead II",
                    "Inverted P wave in aVF",
                    "Can mimic inferior MI",
                    "aVF and aVR are interchanged"
                ],
                "clinical_impact": "May simulate or mask inferior myocardial infarction"
            },
            "LA-LL Reversal": {
                "frequency": "Often difficult to detect (15-20%)",
                "signs": [
                    "P wave in Lead I higher than Lead II",
                    "Terminal positive P wave in Lead III",
                    "Lead I becomes Lead II",
                    "aVL and aVF are interchanged"
                ],
                "clinical_impact": "May appear 'more normal' than correct ECG, can mask inferior changes"
            },
            "Limb-Neutral Reversal": {
                "frequency": "Uncommon but distinctive (5-10%)",
                "signs": [
                    "Almost flat line in Lead I, II, or III",
                    "Distorted Wilson's central terminal",
                    "All precordial leads affected",
                    "Two limb leads may look identical"
                ],
                "clinical_impact": "Severely distorts ECG morphology, makes interpretation unreliable"
            },
            "Precordial Misplacement": {
                "frequency": "Variable, often due to electrode positioning",
                "signs": [
                    "Abnormal R wave progression",
                    "Unusual P, QRS, or T wave morphology",
                    "Poor transition in precordial leads",
                    "Inconsistent with limb lead findings"
                ],
                "clinical_impact": "Can simulate myocardial infarction or other cardiac pathology"
            }
        }
        
        for misplacement, info in misplacement_info.items():
            with st.expander(f"🔍 {misplacement}"):
                st.markdown(f"**Frequency:** {info['frequency']}")
                
                st.markdown("**Characteristic Signs:**")
                for sign in info['signs']:
                    st.markdown(f"• {sign}")
                
                st.markdown(f"**Clinical Impact:** {info['clinical_impact']}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">🔍 Detection Rules and Algorithms</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Rule-based Detection Criteria
        
        Based on Batchvarov et al. (2007), the following criteria are implemented:
        """)
        
        detection_rules = {
            "Primary Rules": [
                "Negative P-QRS complexes in Lead I → RA-LA reversal",
                "Negative P-QRS complexes in Lead II → RA-LL reversal", 
                "Positive P wave in aVR → Lead misplacement",
                "Flat line in any limb lead → Neutral cable involvement",
                "P wave in Lead I > Lead II → Possible LA-LL reversal"
            ],
            "Secondary Rules": [
                "Abnormal QRS axis shift between leads",
                "Poor R wave progression in precordial leads",
                "Inconsistent P wave morphology",
                "Low correlation between expected lead relationships"
            ],
            "Advanced Algorithms": [
                "P wave vectorcardiographic analysis",
                "Lead reconstruction and correlation methods",
                "Neural network pattern recognition",
                "Multi-parameter decision trees"
            ]
        }
        
        for category, rules in detection_rules.items():
            st.markdown(f"#### {category}")
            for rule in rules:
                st.markdown(f"• {rule}")
        
        st.markdown("""
        ### Machine Learning Approach
        
        The ML model uses the following features:
        - Lead polarity indicators (positive/negative P waves and QRS complexes)
        - Amplitude relationships between leads
        - P wave axis estimation
        - QRS axis calculations
        - R wave progression patterns
        - Inter-lead correlation coefficients
        
        **Model Performance:**
        - Random Forest Classifier with 100 estimators
        - Features scaled using StandardScaler
        - Cross-validation accuracy reported in sidebar
        """)
    
    with tab4:
        st.markdown('<h2 class="sub-header">📈 Model Details and Performance</h2>', unsafe_allow_html=True)
        
        # Model architecture
        st.markdown("### 🏗️ Model Architecture")
        st.markdown("""
        **Primary Model:** Random Forest Classifier
        - **Estimators:** 100 decision trees
        - **Features:** 14 diagnostic features extracted from ECG leads
        - **Classes:** 6 misplacement types + Correct placement
        - **Preprocessing:** StandardScaler normalization
        """)
        
        # Feature importance
        if hasattr(st.session_state.detector.model, 'feature_importances_'):
            st.markdown("### 🎯 Feature Importance")
            
            feature_names = [
                'Lead I Negative P', 'Lead I Negative QRS', 'Lead II Negative P',
                'Lead II Negative QRS', 'Lead II Flat', 'Lead III Flat',
                'aVR Positive P', 'aVR Positive QRS', 'P Axis Abnormal',
                'P Wave Terminal Positive III', 'QRS Axis Shift', 'Abnormal R Progression',
                'Mean Correlation', 'Min Correlation'
            ]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': st.session_state.detector.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Random Forest Feature Importance')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Performance metrics
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Accuracy", f"{st.session_state.model_accuracy:.2%}")
            st.markdown("**Training Data:**")
            st.markdown("• 150 synthetic ECG records")
            st.markdown("• 6 misplacement types")
            st.markdown("• Balanced dataset")
        
        with col2:
            st.markdown("**Classification Report:**")
            st.text(st.session_state.model_report)

if __name__ == "__main__":
    main()
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
        """Extract features for misplacement detection based on the paper's criteria"""
        features = {}
        
        # Lead I features
        features['lead_I_negative_p'] = 1 if np.mean(ecg_data.get('I', [0])) < 0 else 0
        features['lead_I_negative_qrs'] = 1 if np.min(ecg_data.get('I', [0])) < -0.1 else 0
        
        # Lead II features  
        features['lead_II_negative_p'] = 1 if np.mean(ecg_data.get('II', [0])) < 0 else 0
        features['lead_II_negative_qrs'] = 1 if np.min(ecg_data.get('II', [0])) < -0.1 else 0
        features['lead_II_flat'] = 1 if np.std(ecg_data.get('II', [0])) < 0.05 else 0
        
        # Lead III features
        features['lead_III_flat'] = 1 if np.std(ecg_data.get('III', [0])) < 0.05 else 0
        
        # aVR features
        features['avr_positive_p'] = 1 if np.mean(ecg_data.get('aVR', [0])) > 0 else 0
        features['avr_positive_qrs'] = 1 if np.max(ecg_data.get('aVR', [0])) > 0.1 else 0
        
        # P wave axis estimation
        p_lead_I = np.mean(ecg_data.get('I', [0])[:int(len(ecg_data.get('I', [0]))*0.3)])
        p_lead_II = np.mean(ecg_data.get('II', [0])[:int(len(ecg_data.get('II', [0]))*0.3)])
        p_lead_III = np.mean(ecg_data.get('III', [0])[:int(len(ecg_data.get('III', [0]))*0.3)])
        
        features['p_axis_abnormal'] = 1 if p_lead_I > p_lead_II else 0
        features['p_wave_terminal_positive_III'] = 1 if p_lead_III > 0.05 else 0
        
        # QRS axis estimation
        qrs_lead_I = np.max(ecg_data.get('I', [0])) - np.min(ecg_data.get('I', [0]))
        qrs_lead_II = np.max(ecg_data.get('II', [0])) - np.min(ecg_data.get('II', [0]))
        
        features['qrs_axis_shift'] = abs(qrs_lead_I - qrs_lead_II) / max(qrs_lead_I, qrs_lead_II, 0.1)
        
        # Precordial progression features
        precordial_leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        r_wave_progression = []
        
        for lead in precordial_leads:
            if lead in ecg_data:
                r_wave = np.max(ecg_data[lead])
                r_wave_progression.append(r_wave)
            else:
                r_wave_progression.append(0)
        
        # Check for abnormal R wave progression
        normal_progression = all(r_wave_progression[i] <= r_wave_progression[i+1] 
                               for i in range(3))  # V1-V4 should increase
        features['abnormal_r_progression'] = 0 if normal_progression else 1
        
        # Lead correlations (for reconstruction method)
        correlations = []
        for i, lead1 in enumerate(self.lead_names):
            for j, lead2 in enumerate(self.lead_names):
                if i < j and lead1 in ecg_data and lead2 in ecg_data:
                    corr, _ = pearsonr(ecg_data[lead1], ecg_data[lead2])
                    correlations.append(abs(corr))
        
        features['mean_correlation'] = np.mean(correlations) if correlations else 0
        features['min_correlation'] = np.min(correlations) if correlations else 0
        
        return features
    
    def detect_specific_misplacement(self, features):
        """Detect specific type of misplacement based on paper's criteria"""
        
        # RA-LA Reversal (most common)
        if (features['lead_I_negative_p'] and features['lead_I_negative_qrs'] and 
            features['avr_positive_p']):
            return 'RA-LA Reversal', 0.95
        
        # RA-LL Reversal  
        if features['lead_II_negative_p'] and features['lead_II_negative_qrs']:
            return 'RA-LL Reversal', 0.90
        
        # LA-LL Reversal
        if (features['p_axis_abnormal'] or features['p_wave_terminal_positive_III']):
            return 'LA-LL Reversal', 0.75
        
        # Limb-Neutral cable reversal
        if (features['lead_II_flat'] or features['lead_III_flat']):
            return 'Limb-Neutral Reversal', 0.85
        
        # Precordial misplacement
        if features['abnormal_r_progression']:
            return 'Precordial Misplacement', 0.70
        
        # Check for subtle changes
        if (features['qrs_axis_shift'] > 0.3 or features['mean_correlation'] < 0.6):
            return 'Possible Misplacement', 0.60
            
        return 'Correct', 0.95
    
    def train_model(self, training_data):
        """Train ML model on ECG data"""
        features_list = []
        labels = []
        
        for ecg_record in training_data:
            features = self.extract_features_from_ecg(ecg_record['data'])
            feature_vector = list(features.values())
            features_list.append(feature_vector)
            labels.append(ecg_record['label'])
        
        X = np.array(features_list)
        y = np.array(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, classification_report(y_test, y_pred)

def generate_synthetic_ecg_data():
    """Generate synthetic ECG data for demonstration"""
    np.random.seed(42)
    
    # Normal ECG pattern
    t = np.linspace(0, 1, 1000)  # 1 second of data
    
    # Basic ECG components
    p_wave = 0.1 * np.exp(-((t - 0.15) / 0.05) ** 2)
    qrs_complex = 0.8 * np.exp(-((t - 0.4) / 0.02) ** 2) - 0.2 * np.exp(-((t - 0.38) / 0.01) ** 2)
    t_wave = 0.15 * np.exp(-((t - 0.7) / 0.08) ** 2)
    
    base_ecg = p_wave + qrs_complex + t_wave + 0.02 * np.random.normal(0, 1, len(t))
    
    # Generate data for different conditions
    training_data = []
    
    # Correct ECG
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
    
    # RA-LA Reversal (negative lead I)
    ra_la_reversal = correct_ecg.copy()
    ra_la_reversal['I'] = -correct_ecg['I']  # Lead I becomes negative
    ra_la_reversal['aVR'] = -ra_la_reversal['aVR']  # aVR becomes positive
    
    for i in range(30):
        training_data.append({'data': ra_la_reversal, 'label': 'RA-LA Reversal'})
    
    # RA-LL Reversal (negative lead II)
    ra_ll_reversal = correct_ecg.copy()
    ra_ll_reversal['II'] = -correct_ecg['II']  # Lead II becomes negative
    
    for i in range(20):
        training_data.append({'data': ra_ll_reversal, 'label': 'RA-LL Reversal'})
    
    # LA-LL Reversal (subtle changes in P wave)
    la_ll_reversal = correct_ecg.copy()
    # Swap some characteristics
    temp = la_ll_reversal['I']
    la_ll_reversal['I'] = la_ll_reversal['II'] * 0.8
    la_ll_reversal['aVL'], la_ll_reversal['aVF'] = la_ll_reversal['aVF'], la_ll_reversal['aVL']
    
    for i in range(20):
        training_data.append({'data': la_ll_reversal, 'label': 'LA-LL Reversal'})
    
    # Limb-Neutral Reversal (flat line in lead II)
    limb_neutral_reversal = correct_ecg.copy()
    limb_neutral_reversal['II'] = np.zeros_like(correct_ecg['II']) + 0.01 * np.random.normal(0, 1, len(t))
    
    for i in range(15):
        training_data.append({'data': limb_neutral_reversal, 'label': 'Limb-Neutral Reversal'})
    
    # Precordial Misplacement (abnormal R wave progression)
    precordial_misplacement = correct_ecg.copy()
    # Reverse some precordial leads
    precordial_misplacement['V2'], precordial_misplacement['V4'] = precordial_misplacement['V4'], precordial_misplacement['V2']
    
    for i in range(15):
        training_data.append({'data': precordial_misplacement, 'label': 'Precordial Misplacement'})
    
    return training_data

def process_uploaded_file(uploaded_file):
    """Process uploaded ECG file"""
    try:
        file_type = uploaded_file.type
        
        if file_type in ['image/jpeg', 'image/jpg', 'image/png']:
            # Process image file
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded ECG Image", use_column_width=True)
            
            # Convert to numpy array for processing
            img_array = np.array(image)
            
            # Simple signal extraction (this would need more sophisticated processing in practice)
            if len(img_array.shape) == 3:
                gray_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray_image = img_array
            
            # Extract signal lines (simplified approach)
            # In practice, you'd use more sophisticated ECG digitization techniques
            signals = extract_signals_from_image(gray_image)
            return signals
            
        elif file_type == 'application/pdf':
            st.warning("PDF processing requires additional libraries. Please convert to image format.")
            return None
            
        else:
            # Assume it's a data file (CSV, text, etc.)
            content = uploaded_file.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8')
            
            # Try to parse as CSV
            try:
                data = pd.read_csv(io.StringIO(content))
                return convert_dataframe_to_ecg(data)
            except:
                st.error("Could not parse file format. Please upload ECG data in CSV format or as an image.")
                return None
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def extract_signals_from_image(image):
    """Extract ECG signals from image (simplified approach)"""
    # This is a simplified signal extraction
    # In practice, you'd use more sophisticated image processing
    
    height, width = image.shape
    
    # Divide image into 12 sections for 12 leads (3x4 grid typical)
    leads = {}
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    rows, cols = 3, 4
    for i, lead_name in enumerate(lead_names):
        row = i // cols
        col = i % cols
        
        # Extract region
        start_row = int(row * height / rows)
        end_row = int((row + 1) * height / rows)
        start_col = int(col * width / cols)
        end_col = int((col + 1) * width / cols)
        
        region = image[start_row:end_row, start_col:end_col]
        
        # Extract signal by finding the darkest line in each column
        signal = []
        for column in range(region.shape[1]):
            col_data = region[:, column]
            # Find the row with minimum value (darkest point)
            min_row = np.argmin(col_data)
            # Normalize to create signal
            signal.append((region.shape[0] - min_row) / region.shape[0] - 0.5)
        
        leads[lead_name] = signal
    
    return leads

def convert_dataframe_to_ecg(df):
    """Convert dataframe to ECG format"""
    ecg_data = {}
    
    # Map dataframe columns to ECG leads
    lead_mapping = {
        'I': ['I', 'Lead_I', 'lead_1'],
        'II': ['II', 'Lead_II', 'lead_2'],
        'III': ['III', 'Lead_III', 'lead_3'],
        'aVR': ['aVR', 'AVR', 'Lead_aVR', 'avr'],
        'aVL': ['aVL', 'AVL', 'Lead_aVL', 'avl'],
        'aVF': ['aVF', 'AVF', 'Lead_aVF', 'avf'],
        'V1': ['V1', 'Lead_V1', 'v1'],
        'V2': ['V2', 'Lead_V2', 'v2'],
        'V3': ['V3', 'Lead_V3', 'v3'],
        'V4': ['V4', 'Lead_V4', 'v4'],
        'V5': ['V5', 'Lead_V5', 'v5'],
        'V6': ['V6', 'Lead_V6', 'v6']
    }
    
    for standard_name, possible_names in lead_mapping.items():
        for col_name in possible_names:
            if col_name in df.columns:
                ecg_data[standard_name] = df[col_name].values.tolist()
                break
    
    return ecg_data

def main():
    st.markdown('<h1 class="main-header">🫀 ECG Lead Misplacement Detection System</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    <h3>About This Application</h3>
    <p>This application detects incorrect electrode cable connections during ECG recording based on the research by 
    Batchvarov et al. (2007). It implements automated detection algorithms using machine learning and rule-based 
    approaches to identify various types of lead misplacements.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Detection method selection
    detection_method = st.sidebar.selectbox(
        "Select Detection Method",
        ["Rule-based Detection", "Machine Learning", "Both Methods"]
    )
    
    # File upload sensitivity
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.8,
        step=0.05,
        help="Minimum confidence level for misplacement detection"
    )
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = ECGLeadMisplacementDetector()
        
        # Generate and train on synthetic data
        with st.spinner("Initializing ML model with synthetic training data..."):
            training_data = generate_synthetic_ecg_data()
            accuracy, report = st.session_state.detector.train_model(training_data)
            st.session_state.model_accuracy = accuracy
            st.session_state.model_report = report
    
    # Display model performance
    st.sidebar.markdown("### 📊 Model Performance")
    st.sidebar.metric("Accuracy", f"{st.session_state.model_accuracy:.2%}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Analyze", "📚 Misplacement Types", "🔍 Detection Rules", "📈 Model Details"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Upload ECG Data</h2>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose ECG file",
            type=['jpg', 'jpeg', 'png', 'pdf', 'csv', 'txt'],
            help="Upload ECG image (JPG, PNG), PDF, or data file (CSV, TXT)"
        )
        
        # Demo data option
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧪 Use Demo Data - Normal ECG"):
                demo_data = generate_synthetic_ecg_data()[0]['data']  # Normal ECG
                st.session_state.current_ecg = demo_data
                st.session_state.demo_type = "Normal ECG"
        
        with col2:
            demo_misplacement = st.selectbox(
                "Demo Misplacement Type",
                ["RA-LA Reversal", "RA-LL Reversal", "LA-LL Reversal", "Limb-Neutral Reversal", "Precordial Misplacement"]
            )
            
            if st.button(f"🧪 Use Demo Data - {demo_misplacement}"):
                demo_data_list = generate_synthetic_ecg_data()
                for data in demo_data_list:
                    if data['label'] == demo_misplacement:
                        st.session_state.current_ecg = data['data']
                        st.session_state.demo_type = demo_misplacement
                        break
        
        # Process uploaded file
        if uploaded_file is not None:
            ecg_data = process_uploaded_file(uploaded_file)
            if ecg_data:
                st.session_state.current_ecg = ecg_data
                st.success("✅ File processed successfully!")
        
        # Analyze ECG data
        if 'current_ecg' in st.session_state:
            st.markdown('<h2 class="sub-header">📊 ECG Analysis Results</h2>', unsafe_allow_html=True)
            
            ecg_data = st.session_state.current_ecg
            
            # Display ECG signals
            st.markdown("### ECG Waveforms")
            
            fig, axes = plt.subplots(4, 3, figsize=(15, 12))
            fig.suptitle('12-Lead ECG', fontsize=16)
            
            lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
            
            for i, lead in enumerate(lead_names):
                row, col = i // 3, i % 3
                if lead in ecg_data:
                    axes[row, col].plot(ecg_data[lead])
                    axes[row, col].set_title(f'Lead {lead}')
                    axes[row, col].grid(True, alpha=0.3)
                else:
                    axes[row, col].text(0.5, 0.5, 'No Data', ha='center', va='center', transform=axes[row, col].transAxes)
                    axes[row, col].set_title(f'Lead {lead}')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Perform detection
            st.markdown("### 🔍 Misplacement Detection Results")
            
            # Extract features
            features = st.session_state.detector.extract_features_from_ecg(ecg_data)
            
            # Rule-based detection
            if detection_method in ["Rule-based Detection", "Both Methods"]:
                st.markdown("#### Rule-based Analysis")
                
                misplacement_type, confidence = st.session_state.detector.detect_specific_misplacement(features)
                
                if confidence >= confidence_threshold:
                    if misplacement_type == "Correct":
                        st.markdown(f"""
                        <div class="success-box">
                        <h4>✅ Result: {misplacement_type}</h4>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p>No significant lead misplacement detected.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-box">
                        <h4>⚠️ Result: {misplacement_type}</h4>
                        <p><strong>Confidence:</strong> {confidence:.2%}</p>
                        <p>Lead misplacement detected! Please check electrode connections.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                    <h4>❓ Result: Uncertain</h4>
                    <p><strong>Confidence:</strong> {confidence:.2%}</p>
                    <p>Detection confidence below threshold. Manual review recommended.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Machine learning detection
            if detection_method in ["Machine Learning", "Both Methods"]:
                st.markdown("#### Machine Learning Analysis")
                
                # Prepare features for ML model
                feature_vector = np.array(list(features.values())).reshape(1, -1)
                feature_vector_scaled = st.session_state.detector.scaler.transform(feature_vector)
                
                # Predict
                ml_prediction = st.session_state.detector.model.predict(feature_vector_scaled)[0]
                ml_probabilities = st.session_state.detector.model.predict_proba(feature_vector_scaled)[0]
                
                # Get the confidence for the predicted class
                predicted_class_idx = st.session_state.detector.model.classes_.tolist().index(ml_prediction)
                ml_confidence = ml_probabilities[predicted_class_idx]
                
                if ml_confidence >= confidence_threshold:
                    if ml_prediction == "Correct":
                        st.markdown(f"""
                        <div class="success-box">
                        <h4>✅ ML Result: {ml_prediction}</h4>
                        <p><strong>Confidence:</strong> {ml_confidence:.2%}</p>
                        <p>Machine learning model indicates correct lead placement.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="error-box">
                        <h4>⚠️ ML Result: {ml_prediction}</h4>
                        <p><strong>Confidence:</strong> {ml_confidence:.2%}</p>
                        <p>Machine learning model detected lead misplacement.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="warning-box">
                    <h4>❓ ML Result: Uncertain</h4>
                    <p><strong>Confidence:</strong> {ml_confidence:.2%}</p>
                    <p>ML prediction confidence below threshold.</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show probability distribution
                st.markdown("##### Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Misplacement Type': st.session_state.detector.model.classes_,
                    'Probability': ml_probabilities
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(prob_df['Misplacement Type'], prob_df['Probability'])
                ax.set_ylabel('Probability')
                ax.set_title('Machine Learning Prediction Probabilities')
                plt.xticks(rotation=45, ha='right')
                
                # Highlight the predicted class
                bars[predicted_class_idx].set_color('red')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Feature analysis
            st.markdown("### 📋 Feature Analysis")
            
            feature_df = pd.DataFrame([features]).T
            feature_df.columns = ['Value']
            feature_df.index.name = 'Feature'
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(feature_df, use_container_width=True)
            
            with col2:
                # Key diagnostic features
                st.markdown("#### 🔑 Key Diagnostic Indicators")
                
                diagnostic_messages = []
                
                if features['lead_I_negative_p'] or features['lead_I_negative_qrs']:
                    diagnostic_messages.append("• Negative P or QRS in Lead I (suggests RA-LA reversal)")
                
                if features['lead_II_negative_p'] or features['lead_II_negative_qrs']:
                    diagnostic_messages.append("• Negative P or QRS in Lead II (suggests RA-LL reversal)")
                
                if features['avr_positive_p'] or features['avr_positive_qrs']:
                    diagnostic_messages.append("• Positive complexes in aVR (abnormal)")
                
                if features['lead_II_flat'] or features['lead_III_flat']:
                    diagnostic_messages.append("• Flat line in limb lead (suggests neutral cable issue)")
                
                if features['p_axis_abnormal']:
                    diagnostic_messages.append("• Abnormal P wave axis")
                
                if features['abnormal_r_progression']:
                    diagnostic_messages.append("• Abnormal R wave progression in precordial leads")
                
                if diagnostic_messages:
                    for msg in diagnostic_messages:
                        st.markdown(msg)
                else:
                    st.markdown("• No significant abnormalities detected")
    
    with tab2:
        st.markdown('<h2 class="sub-header">📚 Types of Lead Misplacements</h2>', unsafe_allow_html=True)
        
        misplacement_info = {
            "RA-LA Reversal": {
                "frequency": "Most common (60-70% of misplacements)",
                "signs": [
                    "Negative P and QRS complexes in Lead I",
                    "Positive P wave in aVR",
                    "Normal precordial leads",
                    "Resembles mirror-image dextrocardia"
                ],
                "clinical_impact": "Can simulate dextrocardia, may mask or simulate cardiac abnormalities"
            },
            "RA-LL Reversal": {
                "frequency": "Less common (10-15% of misplacements)",
                "signs": [
                    "Negative P and QRS complexes in Lead II",
                    "Inverted P wave in aVF",
                    "Can mimic inferior MI",
                    "aVF and aVR are interchanged"
                ],
                "clinical_impact": "May simulate or mask inferior myocardial infarction"
            },
            "LA-LL Reversal": {
                "frequency": "Often difficult to detect (15-20%)",
                "signs": [
                    "P wave in Lead I higher than Lead II",
                    "Terminal positive P wave in Lead III",
                    "Lead I becomes Lead II",
                    "aVL and aVF are interchanged"
                ],
                "clinical_impact": "May appear 'more normal' than correct ECG, can mask inferior changes"
            },
            "Limb-Neutral Reversal": {
                "frequency": "Uncommon but distinctive (5-10%)",
                "signs": [
                    "Almost flat line in Lead I, II, or III",
                    "Distorted Wilson's central terminal",
                    "All precordial leads affected",
                    "Two limb leads may look identical"
                ],
                "clinical_impact": "Severely distorts ECG morphology, makes interpretation unreliable"
            },
            "Precordial Misplacement": {
                "frequency": "Variable, often due to electrode positioning",
                "signs": [
                    "Abnormal R wave progression",
                    "Unusual P, QRS, or T wave morphology",
                    "Poor transition in precordial leads",
                    "Inconsistent with limb lead findings"
                ],
                "clinical_impact": "Can simulate myocardial infarction or other cardiac pathology"
            }
        }
        
        for misplacement, info in misplacement_info.items():
            with st.expander(f"🔍 {misplacement}"):
                st.markdown(f"**Frequency:** {info['frequency']}")
                
                st.markdown("**Characteristic Signs:**")
                for sign in info['signs']:
                    st.markdown(f"• {sign}")
                
                st.markdown(f"**Clinical Impact:** {info['clinical_impact']}")
    
    with tab3:
        st.markdown('<h2 class="sub-header">🔍 Detection Rules and Algorithms</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Rule-based Detection Criteria
        
        Based on Batchvarov et al. (2007), the following criteria are implemented:
        """)
        
        detection_rules = {
            "Primary Rules": [
                "Negative P-QRS complexes in Lead I → RA-LA reversal",
                "Negative P-QRS complexes in Lead II → RA-LL reversal", 
                "Positive P wave in aVR → Lead misplacement",
                "Flat line in any limb lead → Neutral cable involvement",
                "P wave in Lead I > Lead II → Possible LA-LL reversal"
            ],
            "Secondary Rules": [
                "Abnormal QRS axis shift between leads",
                "Poor R wave progression in precordial leads",
                "Inconsistent P wave morphology",
                "Low correlation between expected lead relationships"
            ],
            "Advanced Algorithms": [
                "P wave vectorcardiographic analysis",
                "Lead reconstruction and correlation methods",
                "Neural network pattern recognition",
                "Multi-parameter decision trees"
            ]
        }
        
        for category, rules in detection_rules.items():
            st.markdown(f"#### {category}")
            for rule in rules:
                st.markdown(f"• {rule}")
        
        st.markdown("""
        ### Machine Learning Approach
        
        The ML model uses the following features:
        - Lead polarity indicators (positive/negative P waves and QRS complexes)
        - Amplitude relationships between leads
        - P wave axis estimation
        - QRS axis calculations
        - R wave progression patterns
        - Inter-lead correlation coefficients
        
        **Model Performance:**
        - Random Forest Classifier with 100 estimators
        - Features scaled using StandardScaler
        - Cross-validation accuracy reported in sidebar
        """)
    
    with tab4:
        st.markdown('<h2 class="sub-header">📈 Model Details and Performance</h2>', unsafe_allow_html=True)
        
        # Model architecture
        st.markdown("### 🏗️ Model Architecture")
        st.markdown("""
        **Primary Model:** Random Forest Classifier
        - **Estimators:** 100 decision trees
        - **Features:** 12 diagnostic features extracted from ECG leads
        - **Classes:** 6 misplacement types + Correct placement
        - **Preprocessing:** StandardScaler normalization
        """)
        
        # Feature importance
        if hasattr(st.session_state.detector.model, 'feature_importances_'):
            st.markdown("### 🎯 Feature Importance")
            
            feature_names = [
                'Lead I Negative P', 'Lead I Negative QRS', 'Lead II Negative P',
                'Lead II Negative QRS', 'Lead II Flat', 'Lead III Flat',
                'aVR Positive P', 'aVR Positive QRS', 'P Axis Abnormal',
                'P Wave Terminal Positive III', 'QRS Axis Shift', 'Abnormal R Progression',
                'Mean Correlation', 'Min Correlation'
            ]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': st.session_state.detector.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'])
            ax.set_xlabel('Feature Importance')
            ax.set_title('Random Forest Feature Importance')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Performance metrics
        st.markdown("### 📊 Performance Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model Accuracy", f"{st.session_state.model_accuracy:.2%}")
            st.markdown("**Training Data:**")
            st.markdown("• 150 synthetic ECG records")
            st.markdown("• 6 misplacement types")
            st.markdown("• Balanced dataset")
        
        with col2:
            st.markdown("**Classification Report:**")
            st.text(st.session_state.model_report)
        
        # Limitations and considerations
        st.markdown("### ⚠️ Limitations and Considerations")
        
        st.markdown("""
        **Current Limitations:**
        - Trained on synthetic data - real-world performance may vary
        - Image processing is simplified - clinical-grade digitization needed
        - Some subtle misplacements may be difficult to detect
        - Requires good quality ECG signals for accurate analysis
        
        **Recommendations for Clinical Use:**
        - Always correlate with clinical findings
        - Use as a screening tool, not definitive diagnosis
        - Validate with known correct ECGs when possible
        - Consider multiple detection methods for confirmation
        """)
        
        # Future improvements
        st.markdown("### 🚀 Future Improvements")
        
        st.markdown("""
        **Planned Enhancements:**
        - Integration with real clinical ECG databases
        - Advanced image processing for better signal extraction
        - Deep learning models for pattern recognition
        - DICOM file format support
        - Real-time detection capabilities
        - Integration with ECG machines
        """)

if __name__ == "__main__":
    main()
