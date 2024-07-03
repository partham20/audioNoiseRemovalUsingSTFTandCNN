
# Audio Noise Removal Using STFT and CNN

This project implements a deep learning-based approach for removing noise from audio files. It uses a Convolutional Neural Network (CNN) model to process the spectrogram of noisy audio and generate a cleaned version.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Features

- Remove background noise from audio files
- Process multiple audio files in batch
- Generate spectrograms and waveforms for input and output audio
- Support for various audio formats (WAV, MP3, etc.)

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- SciPy
- Matplotlib
- Librosa
- tqdm

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/audio-noise-removal.git
   cd audio-noise-removal
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Download the pre-trained model:
   ```
   # Add instructions for downloading the model file
   ```

## Usage

### Single File Processing

To process a single audio file:

```python
from modules import process_audio

input_file = "path/to/your/noisy_audio.wav"
output_dir = "path/to/output/directory"

process_audio(input_file, output_dir)
```

### Batch Processing

To process multiple audio files:

```python
import os
from modules import process_audio

input_dir = "path/to/noisy/audio/files"
output_dir = "path/to/output/directory"

for file_name in os.listdir(input_dir):
    if file_name.endswith(".wav"):
        input_file = os.path.join(input_dir, file_name)
        process_audio(input_file, output_dir)
```

## Model Architecture

The noise removal model is a Convolutional Neural Network (CNN) with the following architecture:

- Input layer: 128x128x1 (spectrogram)
- Multiple convolutional and pooling layers
- Skip connections for better feature preservation
- Output layer: 128x128x1 (cleaned spectrogram)

For more details, refer to the `CNNmodel()` function in `modules.py`.

## Training

The model is trained on pairs of noisy and clean audio samples. To train the model:

1. Prepare your dataset of noisy and clean audio pairs.
2. Update the data loading and preprocessing steps in `modules.py`.
3. Run the training script:
   ```
   python train.py
   ```

Training parameters can be adjusted in the `train.py` file.

## Evaluation

To evaluate the model's performance:

1. Prepare a test set of noisy audio files.
2. Run the evaluation script:
   ```
   python evaluate.py --test_dir path/to/test/files --output_dir path/to/output
   ```

This will process the test files and generate cleaned versions along with spectrograms and waveform visualizations.

## Results

The model's performance can be assessed by:

1. Listening to the original noisy audio and the processed clean audio.
2. Comparing the spectrograms of the input and output.
3. Calculating objective metrics such as Signal-to-Noise Ratio (SNR) improvement.

Example results and visualizations can be found in the `results` directory.

## Contributing

Contributions to this project are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to your fork
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

