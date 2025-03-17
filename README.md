
# WAVELLDM: Desain dan Pengembangan Model Difusi Laten Ringan dalam Peningkatan Kualitas dan Restorasi Suara Ucapan

Wave Lightweight Latent Diffusion Models (WaveLLDM) adalah model difusi laten berbasis deep learning yang dirancang untuk melakukan proses denoising audio pada perangkat tepi. WaveLLDM bertujuan meningkatkan kualitas dan restorasi suara ucapan dengan efisiensi komputasi yang lebih baik dibandingkan model difusi konvensional.

## 📌 **Latar Belakang**
Model berbasis difusi telah menunjukkan keunggulan dalam sintesis audio, dengan stabilitas pelatihan yang lebih baik dibandingkan Generative Adversarial Networks (GAN) dan model autoregresif. Namun, tantangan utama model ini adalah tingginya kebutuhan daya komputasi. WaveLLDM dikembangkan sebagai solusi dengan arsitektur yang lebih ringan, memungkinkan implementasi pada perangkat tepi seperti Android dan aplikasi web.

## ⚙ **Komponen Neural Audio Codec WaveLLDM**
WaveLLDM terdiri dari tiga komponen utama dalam generator:
1. **ConvNeXt Encoder**: Memproses spektogram mel audio menjadi representasi laten $z$ dengan dimensi laten **384 × L**.
2. **Grouped Residual Finite Scalar Quantization (GFSQ)**: Melakukan kuantisasi pada $\vec{z}$ menjadi $\vec{z_q}$ menggunakan GFSQ. WaveLLDM menggunakan **4 codebook**, dengan level per codebook: **[8, 8, 8, 6, 5]**.
3. **HiFi-GAN Decoder**: Mengembalikan $\vec{z_q}$ menjadi audio asal.

Selain itu, Neural Audio Codec WaveLLDM dilengkapi dengan **HiFi-GAN Discriminator**: Multi-Period Discriminator (MPD) dan Multi-Scale Discriminator (MSD).

## 🎯 **Loss Functions**
Model ini dilatih menggunakan kombinasi dari empat fungsi loss:
1. **Adversarial Loss**
2. **Feature Matching Loss** (Lambda = **2**)
3. **Multiscale Mel Spectrogram Loss** (Lambda = **65**)
4. **Multiscale STFT Spectral Loss** (Lambda = **20**)

## 🔨 **Arsitektur Diffusion (Coming Soon)**
Selanjutnya, akan dibangun model **U-Net** yang bekerja di **latent space** untuk melakukan difusi laten. Arsitektur U-Net ini akan didukung dengan **Causal Conv 1D** dan **Rotary Attention**, yang membantu dalam memahami konteks sekuensial audio.

## 📂 **Dataset**
- **Training:** Voicebank+DEMAND dan dataset terkait lainnya.
- **Evaluation:** LJSpeech dengan metrik **Mean Opinion Score (MOS)**.

## 🚀 **Instalasi dan Penggunaan**
### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Training Model**
Gunakan skrip `train_vctk.py` dengan parameter berikut (sesuaikan jika diperlukan):
```bash
python train_vctk.py \
  --clean_data_path ./vctk/clean_trainset_56spk_wav/clean_trainset_56spk_wav/ \
  --noisy_data_path ./vctk/noisy_trainset_56spk_wav/noisy_trainset_56spk_wav/ \
  --add_random_cutting True \
  --batch_size 16 \
  --lr 2e-5 \
  --use_scheduler True \
  --epochs 100 \
  --num_workers 16
```

## ✅ **Progress Saat Ini**
✔ Perancangan Neural Audio Codec  
✔ Perancangan Arsitektur U-Net  
✔ Training Cycle  
✔ Loss Functions, Dataset, Data Loader, dan modul pendukung lainnya  

## ⏳ **To-Do List**
❌ Melatih U-Net untuk DDPM/DDIM  
❌ Pengujian end-to-end  
❌ Implementasi kode inferensi  
❌ Implementasi UI dengan **Gradio**  

## 📌 **Rencana Implementasi**
Setelah model difusi selesai, WaveLLDM akan diimplementasikan dalam:
- **Aplikasi Web** untuk demonstrasi real-time.
- **Perangkat Tepi (Android)** untuk denoising suara langsung dari perangkat.

## 📄 **Lisensi**
Lisensi akan ditentukan nanti.
