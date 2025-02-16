import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def plot_signal(title, file_name, flag):
    plt.figure(figsize=(10, 6))

    if flag == 'noisy':
        for col in signal_data.columns:
            combined_signal = signal_data.mean(axis=1)
            plt.plot(combined_signal, label=col)
    elif flag == 'smooth':
        for col in signal_data.columns:
            combined_signal = signal_data.mean(axis=1)
            signal_smoothed = savgol_filter(combined_signal, window_length=11, polyorder=2)
            plt.plot(signal_smoothed, label=col)
    
    plt.xlabel('Wavelengths')
    plt.ylabel('Intensity')
    plt.title(title)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')


# In the target variable (class), we chose to drop the unknown values
df = pd.read_csv('data.csv')
df = df[df['class'] != 8]
print(df)

# plotting our signal
filtered_df = df[df['device_id'] == 'B0236F1F2D02C632']
signal_data = filtered_df.iloc[:, 5:]

plt.figure(figsize=(10, 6))


plot_signal('Original signal', 'images/signal_plot_noisy.png', 'noisy')
plot_signal('Smoothed signal', 'images/signal_plot_smooth.png', 'smooth')

