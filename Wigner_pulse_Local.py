# Wigner_pulse_app.py
# calculate the wigner function under the differrent dispersion curves. 

# from PIL import Image
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# # Set title
#st.title("Wigner function calculations for the Pulse (Time & Frequency Domains)")
st.title("Wigner function calculations")

# ---------------------------------
#  Define function
# ---------------------------------
# FFT
def fftc(signal):
    return np.fft.fftshift(np.fft.fft(signal))

def ifftc(signal):
    return np.fft.ifftshift(np.fft.ifft(signal))

def Initial_define():
    # 🕒 time （Unit ：ps）
    t_max = 10  
    n_points = 1025  
    t = np.linspace(-t_max, t_max, n_points)
    dt = t[1] - t[0] 
    W_max=0.5/dt
    W=np.linspace(-W_max, W_max, n_points)
    dW= W[1]-W[0]
    return t,W,dW

def compute_wigner(Ex):
    """
    Calculate Wigner function
    """
    if Ex.ndim != 1:
        raise ValueError("Ex must be a 1D array (column vector in MATLAB)")

    N = len(Ex)
    x = np.fft.ifftshift((np.arange(N) - N / 2) * 2 * np.pi / (N - 1))
    X = np.arange(N) - N / 2

    Ex_fft = np.fft.fft(Ex)
    Ex1 = np.fft.ifft((Ex_fft[:, np.newaxis]) * np.exp(1j * np.outer(x, X / 2)), axis=0)
    Ex2 = np.fft.ifft((Ex_fft[:, np.newaxis]) * np.exp(-1j * np.outer(x, X / 2)), axis=0)

    W_temp = Ex1 * np.conj(Ex2)
    W = np.real(np.fft.fftshift(np.fft.fft(np.fft.fftshift(W_temp, axes=1), axis=1), axes=1))

    return W

# ---------------------------------

t, W,dW = Initial_define()

delt_W=2;
# frequency- 
Ew0=np.exp(-W**2/(2*delt_W**2)) 

# 📌 

GVD_alpha_05 = st.slider("Set GVD0.5 (Unit- ps^0.5)", -10.0, 10.0, 0.0, 0.1, key="gvd_05")
GVD_alpha_1 = st.slider("Set GVD1 (Unit- ps^1)", -10.0, 10.0, 0.0, 0.1, key="gvd_1")
GVD_alpha_2 = st.slider("Set GVD2 (Unit- ps^2)", -10.0, 10.0, 0.0, 0.1, key="gvd_2")
GVD_alpha_3 = st.slider("Set GVD3 (Unit- ps^3)", -10.0, 10.0, 0.0, 0.1, key="gvd_3")  


phase_Dis=GVD_alpha_2*W**2*0.5+GVD_alpha_3*W**3/6+GVD_alpha_1*np.abs(W)**1*0.5+GVD_alpha_05*np.abs(W)**0.5*0.5
Ew=Ew0*np.exp(1j*phase_Dis)

# 📊 fre. analysis
Et=ifftc(Ew)

I_w= np.abs(Ew)**2  # 
I_t= np.abs(Et)**2  # 

# Wigner
Wig =compute_wigner(Ew)
Wig=Wig*dW*1
                         
    col_left, col_right = st.columns([1, 1])  #
    
    # ========= Left+Right =========
    with col_left:
       # st.subheader("Time and Frequency Domain")
        
        # time
        fig1, ax1 = plt.subplots(figsize=(5, 2.5))
        ax1.plot(t, I_t, color='blue')
        ax1.set_xlabel("Time (ps)")
        ax1.set_ylabel("Intensity")
        ax1.set_title("Time Domain")
        ax1.grid(True)
        st.pyplot(fig1)
    
        # # fre.
        fig2, ax2 = plt.subplots(figsize=(5, 2.5))
        ax2.plot(W, I_w, color='red',label="Intensity")
        ax2.set_xlabel("Frequency (THz)")
        ax2.set_ylabel("Spectral Intensity")
        ax2.set_title("Frequency Domain")
        ax2.grid(True)
        
        ax22 = ax2.twinx()
        ax22.plot(W, phase_Dis,color='gray', linestyle='--',label="Phase")
        ax22.set_ylabel("Spectral Phase")
   
        st.pyplot(fig2)
  
    # ========= Wigner =========
    with col_right:
      #  st.subheader("Wigner Distribution")
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        
        extent = [t[0], t[-1], W[0], W[-1]]
        im = ax3.imshow(Wig, extent=extent, origin='lower', aspect='auto', cmap='jet')
        
        ax3.set_xlim([-5, 5])         # 
        ax3.set_ylim([-5, 5])     # 
    
        ax3.set_xlabel("Time (ps)")
        ax3.set_ylabel("Frequency (THz)")
        ax3.set_title("Wigner Distribution")
        plt.colorbar(im, ax=ax3, label="W(t, ω)")
        
        plt.tight_layout()  # 
        
        st.pyplot(fig3)
