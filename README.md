### 環境
env.yaml

### 資料集
* 預訓練：traditional_wang127k.json
* 微調：sighan13/14/15_training_set_traditional.pkl(和在一起用)
* 驗證：sighan13/14/15_test_set_traditional.pkl

### 步驟
1. 執行 start.py 進行預訓練
2. 載入預訓練模型(已寫在腳本中)，執行 sighan_finetuned.py
3. 執行 eval.py 得到 p, r, f 指標結果 
