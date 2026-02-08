# Time-Series Prediction with Elman RNN

This repository demonstrates **time-series forecasting** using **Dense Neural Networks**, **SimpleRNN (Elman RNN)**, and sequence-to-sequence RNNs on a custom multi-sensor dataset.  

The notebook includes:
- Dataset loading and preprocessing from multiple CSV files  
- Correlation analysis and window generation for time-series data  
- Model implementation: Dense network, Elman RNN, and sequence-to-sequence Elman RNN  
- Training, evaluation, and visualization of predictions  



---

## Preprocessing

- **Windowing:** Generates sequences of 80 timesteps as input for prediction  
- **Normalization:** StandardScaler applied to training, validation, and test sets  
- **Train-Test Split:**  
  - 60% training  
  - 15% validation  
  - 25% testing  

---

## Models

### Dense Network
- Predicts the `x` value at the next timestep (t+1) from previous 80 steps  
- Layers: Dense(80) → Dense(79) → Flatten → Dense(1)  

### Elman RNN
- SimpleRNN predicts `x` for the next timestep  
- Layers: SimpleRNN(80) → Dense(79) → Flatten → Dense(1)  

### Sequence-to-Sequence Elman RNN
- Predicts a sequence of 80 future `x` values from previous 80 timesteps  
- Layers: SimpleRNN(80, return_sequences=True) → Dense(80) → Dense(1)  

---

## Training

- Optimizer: `Adam`  
- Loss: `MSE` (Mean Squared Error)  
- Metrics: `MSE`  
- Early stopping on validation loss with patience=5  

---

## Evaluation

- Performance metrics:  
  - Mean Squared Error (MSE)  
  - Mean Absolute Error (MAE)  
- Visual comparison of predictions vs actual data using random samples  

**Example results:**
Elman Neural Network results:
- `images/result_1.png`  
- `images/result_2.png`  

---

## Notes

- Windowing allows prediction from input sequences  
- Sequence-to-sequence RNN achieves the best performance for multi-step forecasting  
- Random samples are used for visual evaluation  

---

## Author

**Mahdieh Nouri**  
📧 [mahdiyenouri99@gmail.com](mailto:mahdiyenouri99@gmail.com)  

---

## License

This project is intended for educational and academic use.
