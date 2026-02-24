# Heart Rate Prediction from Fingertip Video

This project predicts heart rate and SpO2 from fingertip video data using deep learning and machine learning models.
Download this [notebook](https://github.com/AndrewBlur/HeartRatePrediction/blob/main/notebooks/starter_kit.ipynb) and try it in your colab 
## Project Structure

- `dataset/`:  he data is from [MEDVSE repository](https://github.com/MahdiFarvardin/MEDVSE).
- `model/`: Contains the Python source code for data loading, preprocessing, modeling, training, evaluation, and prediction.
- `notebooks/`: Can be used for experimental notebooks.

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download data:**
   - The data is from [MEDVSE repository](https://github.com/MahdiFarvardin/MEDVSE).

3. **Run the pipeline:**
   ```bash
   python main.py
   ```

## Models

This project implements and compares the following models:

- Random Forest
- Extra Trees
- Temporal Convolutional Network (TCN)
