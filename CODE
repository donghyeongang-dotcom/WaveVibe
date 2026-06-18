import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import os
import sys

# ==========================================
# 1. Streamlit 페이지 설정 및 웹 디자인 정의
# ==========================================
st.set_page_config(
    page_title="WaveVibe - 오디오 신호 분석기",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS를 사용하여 현대적이고 고급스러운 다크 테마(Sleek Dark Mode) UI 구축
st.markdown("""
<style>
    /* 웹 폰트 로드 (Outfit & Inter) */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600&display=swap');
    
    /* 전체 애플리케이션의 기본 폰트 설정 */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* 타이틀 및 헤더 전용 폰트 설정 */
    h1, h2, h3, .title-text {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
    }
    
    /* 배경 및 그라디언트 헤더 카드 설정 */
    .hero-container {
        background: linear-gradient(135deg, #1e1e38 0%, #0f0c1b 100%);
        border-radius: 20px;
        padding: 35px;
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        background: linear-gradient(90deg, #FF007F, #7B2CBF, #00F0FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        margin-bottom: 10px;
    }
    
    .hero-subtitle {
        color: #B0B0D0;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* 글래스모피즘 카드 컨테이너 */
    .analysis-card {
        background: rgba(30, 30, 46, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 25px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    
    .analysis-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 240, 255, 0.3);
    }
    
    /* 정보 칩 (Chips) 및 배지 */
    .info-badge {
        display: inline-block;
        background: rgba(123, 44, 191, 0.2);
        color: #C77DFF;
        border: 1px solid rgba(123, 44, 191, 0.4);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    
    /* 키-밸류 메트릭 디자인 */
    .metric-box {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.03);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00F0FF;
        font-family: 'Outfit', sans-serif;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #8E92B2;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Matplotlib 그래프의 전역 스타일에 다크 테마 디자인 적용 (Sleek Dark Theme)
plt.style.use('dark_background')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['text.color'] = '#E2E8F0'
plt.rcParams['axes.labelcolor'] = '#94A3B8'
plt.rcParams['xtick.color'] = '#64748B'
plt.rcParams['ytick.color'] = '#64748B'
plt.rcParams['grid.color'] = '#334155'
plt.rcParams['grid.alpha'] = 0.5

# ==========================================
# 2. 메인 화면 헤더 영역
# ==========================================
st.markdown("""
<div class="hero-container">
    <div class="hero-title">WaveVibe Studio</div>
    <div class="hero-subtitle">파이썬 기반 디지털 신호 처리(DSP) 기술을 활용한 .wav 파일 정밀 분석 시스템</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 3. 사이드바 구성 (파일 업로드 전용)
# ==========================================
st.sidebar.markdown("### 🎙️ 오디오 컨트롤 센터")

uploaded_file = st.sidebar.file_uploader(
    "WAV 오디오 파일 선택", 
    type=["wav"],
    help="16-bit 또는 24-bit PCM 형식의 표준 WAV 파일 업로드를 권장합니다."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**💡 신호 처리 기초 상식**
- **시간 영역 (Time Domain)**: 시간에 따른 공기 압력의 변동(진폭)을 실시간으로 확인하는 가장 직관적인 형태입니다.
- **주파수 영역 (Frequency Domain)**: 오디오 신호에 어떤 주파수(Hz) 성분이 얼마나 강력하게 들어있는지(Amplitude)를 분석합니다. (FFT 활용)
- **스펙트로그램 (Spectrogram)**: 시간에 따라 주파수 분포가 어떻게 변화하는지 보여주는 3차원적 시각화 방식입니다.
""")

# ==========================================
# 4. 파일 분석 및 메인 프로그램 흐름
# ==========================================
if uploaded_file is not None:
    with st.spinner("⏳ 오디오 신호를 로드하고 분석하는 중..."):
        # 오디오 로드 (Librosa 활용)
        y_raw, sr = librosa.load(uploaded_file, sr=None, mono=False)
        audio_bytes = uploaded_file.getvalue()
        
        # 2D 배열인 경우 채널 축이 0번이 되도록 조정 (shape: (2, N) 또는 (N,))
        if y_raw.ndim > 1:
            if y_raw.shape[0] != 2 and y_raw.shape[1] == 2:
                y_raw = y_raw.T
            y = np.mean(y_raw, axis=0)  # 분석용 모노 다운믹스
            duration = y_raw.shape[1] / sr
        else:
            y = y_raw
            duration = len(y) / sr
            
        peak_amplitude = np.max(np.abs(y_raw))

    # 오디오 플레이어 추가 (웹 화면 상단에서 바로 청취 가능)
    st.markdown(f"### 🎧 오디오 플레이어 - **{uploaded_file.name}**")
    st.audio(audio_bytes, format="audio/wav")
    st.write("")

    # 오디오 요약 스펙 카드 노출 (메트릭)
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown("<h4 style='color:#FFF; margin-top:0;'>📊 파일 기술적 상세 명세</h4>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{sr:,} Hz</div>
            <div class="metric-lbl">샘플링 레이트 (Sampling Rate)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{duration:.3f} s</div>
            <div class="metric-lbl">재생 시간 (Duration)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{len(y):,}</div>
            <div class="metric-lbl">총 샘플 개수 (Total Samples)</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-val">{peak_amplitude:.3f}</div>
            <div class="metric-lbl">최대 진폭 (Peak Amplitude)</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------
    # 4-2. GRAPH 1: Time Domain Waveform
    # ------------------------------------------
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown("### 📈 1. Time Domain Waveform (시간 영역 파형)")
    st.markdown("""
    오디오 신호의 기본적인 시간에 따른 공기 압축(진폭)의 흐름입니다.  
    음량이 커지면 진폭이 커지고, 음량이 줄어들면 진폭도 0에 가까워집니다.
    """)
    
    # 시간 벡터 생성
    time_axis = np.linspace(0, duration, num=len(y))
    
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    fig1.patch.set_facecolor('#1E1E2E')
    ax1.set_facecolor('#1A1B26')
    
    # 파형 시각화 (스테레오인 경우 각 채널을 분리하여 시각화)
    if y_raw.ndim > 1:
        ax1.plot(time_axis, y_raw[0], color='#00F0FF', alpha=0.7, linewidth=1.0, label='Left Channel (좌)')
        ax1.plot(time_axis, y_raw[1], color='#FF7E00', alpha=0.7, linewidth=1.0, label='Right Channel (우)')
        ax1.legend(loc='upper right')
    else:
        ax1.plot(time_axis, y, color='#00F0FF', alpha=0.8, linewidth=1.2, label='Mono (모노)')
    ax1.axhline(0, color='white', linestyle='--', alpha=0.3)
    
    ax1.set_title("Time Domain Waveform (시간 영역 파형)", fontsize=12, pad=12, color='#00F0FF', weight='bold')
    ax1.set_xlabel("Time (Seconds)", fontsize=10, labelpad=8)
    ax1.set_ylabel("Amplitude (진폭)", fontsize=10, labelpad=8)
    ax1.set_xlim(0, duration)
    ax1.set_ylim(-1.1, 1.1)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    st.pyplot(fig1)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------
    # 4-3. GRAPH 2: Frequency Spectrum (FFT)
    # ------------------------------------------
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown("### 📊 2. Frequency Spectrum (주파수 스펙트럼)")
    st.markdown("""
    고속 푸리에 변환(FFT) 연산을 통해 시간 영역의 오디오 데이터를 **주파수 성분**으로 완전 분해한 결과입니다.  
    *Nyquist 이론*에 따라 물리적으로 유효한 **절반 주파수 영역(Nyquist Frequency, 샘플링 주파수의 1/2)**까지만 계산하여 분석합니다.
    """)

    N = len(y)
    fft_result = fft(y)
    fft_magnitude = np.abs(fft_result) / N
    fft_magnitude = fft_magnitude[:N//2] * 2
    
    frequencies = fftfreq(N, d=1/sr)
    frequencies = frequencies[:N//2]
    
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    fig2.patch.set_facecolor('#1E1E2E')
    ax2.set_facecolor('#1A1B26')
    
    ax2.plot(frequencies, fft_magnitude, color='#FF007F', alpha=0.9, linewidth=1.0)
    
    ax2.set_title("Frequency Spectrum (주파수 스펙트럼)", fontsize=12, pad=12, color='#FF007F', weight='bold')
    ax2.set_xlabel("Frequency (Hz)", fontsize=10, labelpad=8)
    ax2.set_ylabel("Normalized Amplitude (정규화된 크기)", fontsize=10, labelpad=8)
    ax2.set_xlim(0, sr/2)
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    use_log_freq = st.checkbox("x축 주파수를 로그 스케일(Log Scale)로 변경하여 저주파수 대역을 더 자세히 분석합니다.", value=False)
    if use_log_freq:
        ax2.set_xscale('log')
        ax2.set_xlim(20, sr/2)
        st.caption("ℹ️ 로그 스케일을 적용하면 저주파수(베이스 영역) 대역이 넓게 표현되어 악기 성분 등의 식별이 훨씬 쉬워집니다.")
        
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

    # ------------------------------------------
    # 4-4. GRAPH 3: Spectrogram
    # ------------------------------------------
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown("### 🔮 3. Spectrogram (스펙트로그램)")
    st.markdown("""
    시간(x축)에 따른 주파수 성분(y축)의 에너지 밀도(색상) 변화를 한눈에 관찰할 수 있는 다차원 지도입니다.  
    단시간 푸리에 변환(STFT) 기법을 사용하며, 인간이 인지하는 소리의 크기 감각에 맞춰 **데시벨(dB)** 단위로 변환해 표현합니다.
    """)
    
    stft_data = librosa.stft(y, n_fft=2048, hop_length=512)
    stft_magnitude = np.abs(stft_data)
    stft_db = librosa.amplitude_to_db(stft_magnitude, ref=np.max)
    
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    fig3.patch.set_facecolor('#1E1E2E')
    ax3.set_facecolor('#1A1B26')
    
    img = librosa.display.specshow(
        stft_db, 
        sr=sr, 
        hop_length=512, 
        x_axis='time', 
        y_axis='linear', 
        ax=ax3, 
        cmap='magma'
    )
    
    ax3.set_title("Spectrogram (스펙트로그램 - Linear Scale)", fontsize=12, pad=12, color='#7B2CBF', weight='bold')
    ax3.set_xlabel("Time (Seconds)", fontsize=10, labelpad=8)
    ax3.set_ylabel("Frequency (Hz)", fontsize=10, labelpad=8)
    
    colorbar = fig3.colorbar(img, ax=ax3, format="%+2.0f dB")
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label("Energy (dB)", fontsize=9, labelpad=8)
    
    spec_scale = st.radio(
        "스펙트로그램 y축 주파수 표현 방식 선택:",
        ("Linear (선형)", "Log (로그 - 인간 청각 특성 반영)"),
        horizontal=True
    )
    
    if spec_scale == "Log (로그 - 인간 청각 특성 반영)":
        fig3.clear()
        ax3 = fig3.add_subplot(111)
        img = librosa.display.specshow(
            stft_db, 
            sr=sr, 
            hop_length=512, 
            x_axis='time', 
            y_axis='log', 
            ax=ax3, 
            cmap='magma'
        )
        ax3.set_title("Spectrogram (스펙트로그램 - Log Scale)", fontsize=12, pad=12, color='#7B2CBF', weight='bold')
        ax3.set_xlabel("Time (Seconds)", fontsize=10)
        ax3.set_ylabel("Frequency (Hz - Log Scale)", fontsize=10)
        colorbar = fig3.colorbar(img, ax=ax3, format="%+2.0f dB")
        colorbar.ax.tick_params(labelsize=8)
        colorbar.set_label("Energy (dB)", fontsize=9)
        
    plt.tight_layout()
    st.pyplot(fig3)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ------------------------------------------
    # 4-5. 하단 피드백/가이드 추가 (초보자를 위한 해설)
    # ------------------------------------------
    st.markdown('<div class="analysis-card">', unsafe_allow_html=True)
    st.markdown("### 🎓 분석 결과 해석 가이드 (초보자용)")
    st.markdown("""
    1. **Time Domain Waveform**: 파형의 높낮이가 수직으로 넓게 퍼진 부분은 소리가 큰 구간이고, 가늘게 모인 부분은 무음이거나 매우 조용한 구간입니다.
    2. **Frequency Spectrum**: 그래프가 뾰족하게 솟아오른 주파수 지점(Peak)이 바로 이 음원에서 핵심적으로 울리는 주파수입니다. (예: 저주파수 100~300Hz 대역은 쿵쿵거리는 베이스 드럼이나 무거운 베이스 기타 음, 고주파수 5kHz 이상 대역은 금속 타악기나 쉭쉭거리는 바람 소리 등)
    3. **Spectrogram**: 밝은 노란색/주황색으로 빛나는 영역은 특정 시간에 해당 주파수 성분의 에너지가 극도로 집중되었음을 의미합니다. 보컬의 멜로디 라인이나 악기의 타이밍 변화가 지도로 선명하게 그려집니다.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # 파일을 업로드하지 않았을 때 노출하는 가이드 화면
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px; background: rgba(30,30,46,0.4); border-radius:20px; border: 1px dashed rgba(255,255,255,0.1); margin-top:20px;">
        <span style="font-size: 5rem;">🎵</span>
        <h2 style="color: #FFF; margin-top: 20px; font-weight: 600;">오디오 파일을 기다리고 있습니다!</h2>
        <p style="color: #8E92B2; max-width: 500px; margin: 10px auto 25px auto; font-size: 0.95rem; line-height: 1.6;">
            분석을 시작하려면 좌측 사이드바 패널에서 분석하고자 하는 <b>.wav 형식의 오디오 파일</b>을 업로드해 주세요. 실시간으로 고속 FFT 연산 및 파형 시각화 프로세스가 즉시 실행됩니다.
        </p>
        <div style="display: flex; justify-content: center; gap: 10px; flex-wrap: wrap;">
            <span class="info-badge">Time Domain Waveform</span>
            <span class="info-badge">Frequency Spectrum</span>
            <span class="info-badge">STFT Spectrogram</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
