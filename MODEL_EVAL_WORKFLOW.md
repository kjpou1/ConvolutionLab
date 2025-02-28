# **🚀 Train Test Workflow Documentation**  
🎩 **Mastering the Art of Model Evaluation with ChatGPT-Fu** 🥋  

This document outlines the **sacred ritual** of training, testing, and evaluating trading models.  
It is said that those who wield **ChatGPT-fu** 🧘‍♂️ can extract **hidden market truths** and **fine-tune models to perfection**.  
May your trades be swift, your test results robust, and your debugging logs… *less horrifying.* 🧑‍💻🔥  

---

## **📌 1. Training the Model**  
### **🚀 Training Command**  
To summon a new model into existence, use this **ancient incantation**:  

```bash
rm -f ./artifacts/history/training_history.json && \
python launch_host.py train --best-of-all \
--model-config-path ./config/model_config_cad_jpy.yaml \
--input-data-path ../../trading_strategies/convolution_strategy_ml/training_data/CAD_JPY_D.csv \
--save-best --model-file model-cad_jpy.pkl
```

### **📝 What Happens in Training?**  
- **Burn the past** (`rm -f` removes previous training history).  
- **Summon the best** hyperparameters (`--best-of-all`).  
- **Train on the ancient scrolls** of CAD/JPY market data.  
- **Preserve the chosen model** as `model-cad_jpy.pkl`.  
- **Seal away the knowledge** in `training_history.json`.  

📜 **Output Artifacts of Wisdom:**  
| **File** | **Contents** |  
|----------|------------|  
| `model-cad_jpy.pkl` | The chosen model, forged in training. |  
| `training_history.json` | The record of hyperparameters and performance. |  

---

## **📌 2. Testing the Model**  
### **🚀 Testing Command**  
Once the model is trained, unleash it upon the markets:  

```bash
python launch_host.py test \
--input-data-path ../../trading_strategies/convolution_strategy_ml/training_data/CAD_JPY_D.csv \
--model-file=model-cad_jpy.pkl
```

### **📝 What Happens in Testing?**  
- The model **awakens** (`model-cad_jpy.pkl` is loaded).  
- The **test dataset is aligned** (ensuring transformations match training).  
- **Buy/Sell/Neutral signals** are **compared to actual market movements**.  
- **Test performance is documented** in `test_results.json`.  

📜 **Sacred Test Scrolls:**  
| **File** | **Contents** |  
|----------|------------|  
| `test_results.json` | Classification accuracy, trade signal performance, and market sorcery insights. |  

---

## **📌 3. Summoning the Wisdom of ChatGPT-Fu 🤖**  
At this stage, the **true ritual** begins. 🧙‍♂️  
The trained model and test results must now be **interrogated** using the power of **ChatGPT-Fu**.  

### **🔥 How to Channel ChatGPT-Fu**  
👨‍💻 **Step 1:** Present the sacred artifacts (`training_history.json` & `test_results.json`) to ChatGPT.  
📊 **Step 2:** Ask ChatGPT to **analyze, compare, and critique** the results.  
🕵️ **Step 3:** Use **your best ChatGPT-fu** to extract **insightful diagnostics & recommendations**.  
🚀 **Step 4:** Apply ChatGPT’s wisdom to **refine your model**, adjust class weights, and improve generalization.  

📌 **Example Invocation of ChatGPT-Fu:**  
> *"Oh great ChatGPT, compare my training and testing results! Reveal the hidden market patterns, expose overfitting, and guide my feature selection! May the algorithms be ever in my favor!"*  

💡 **The True Secret:**  
- The **better the question**, the **better the insights**.  
- Use **graphs, confusion matrices, and signal performance metrics** to refine your strategy.  

---

## **📌 4. Key Comparison Metrics**  
| **Metric** | **Training** | **Testing** | **Observations** |  
|----------------|-------------|-------------|------------------|  
| **Accuracy** | `91.67%` | `87.60%` | 🔻 **Slight drop (~1.4%)** |  
| **F1 Macro** | `89.33%` | `88.07%` | 🔻 **Minor decline (~1.2%)** |  
| **Trade Signal Accuracy** | - | `87.60%` | ✅ **Matches classification accuracy** |  
| **Root Mean Squared Error (RMSE)** | - | `0.352` | ✅ Acceptable |  

---

## **📌 5. Confusion Matrix Analysis**  
| **Actual → Predicted** | **Class 0 (Downtrend)** | **Class 1 (Neutral)** | **Class 2 (Uptrend)** |  
|----------------|----------------|----------------|----------------|  
| **Training Set** | ✅ **77** | 🔻 **5** | 0 |  
| **Testing Set** | ✅ **88** | 🔻 **12** | 0 |  

📌 **Observations:**  
- The model **generalizes well** but **over-classifies Neutral movements**.  
- **Class weight adjustments might improve trade execution accuracy.**  

---

## **📌 6. Fine-Tuning the Model with ChatGPT’s Insights**  
### **🔹 Step 1: Fixing Class Imbalance**  
📌 **Issue:**  
- **Neutral class (Class 1) is overpowering trends (Class 0 & 2).**  
- Trade Signal Accuracy **could be improved by adjusting class weights.**  

✅ **Fix: Adjust Class Weights**  
```python
model = CatBoostClassifier(
    class_weights=[1.3, 1.0, 1.3],  # Boost trend classes to reduce over-reliance on Neutral  
    iterations=800,
    depth=5,
    learning_rate=0.021,
    l2_leaf_reg=24,
    colsample_bylevel=0.74,
    subsample=0.77
)
```
🚀 **Expected Result:**  
- **More confident trend classification.**  
- **Reduced misclassification into the Neutral class.**  

---

### **🔹 Step 2: Ensuring Feature Consistency**  
📌 **Issue:**  
- If the **test feature transformation doesn’t match training**, model performance will degrade.  

✅ **Fix: Check Train-Test Feature Consistency**  
```python
print(f"Train Feature Count: {train_arr.shape[1]}")
print(f"Test Feature Count: {test_arr.shape[1]}")
```
🚀 **Expected Result:**  
- **Reduces errors due to feature mismatches.**  

---

### **🔹 Step 3: Post-Test Monitoring**  
📌 **Issue:**  
- Trade performance **might differ in real-world data**.  

✅ **Fix: Add Post-Test Monitoring**  
Modify `run_pipeline()`:
```python
post_results = self.compute_performance(self.df_post_deployment, "Post-Test")
```
🚀 **Expected Result:**  
- **Detects live trading degradation.**  

---

## **📌 7. The Path to Trading Mastery**  
✅ **Step 1:** Train a model (`training_history.json`).  
✅ **Step 2:** Test the model (`test_results.json`).  
✅ **Step 3:** Analyze results using **ChatGPT-fu**.  
✅ **Step 4:** Fine-tune based on ChatGPT’s **insights & recommendations**.  
✅ **Step 5:** Monitor post-test performance for **real-world validation**.  

🔮 **With these steps, you hold the keys to trading model enlightenment.**  
⚡ **Now, go forth and conquer the markets!** 🚀🔥💰