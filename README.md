# RentAI_STGCN  
### 應用政府開放資料評估土地開發潛力-以桃園市水利管理處精華土地為例
資產活化AI應用創新競賽--第二名  
(以下敘述著重在AI部分)
## 計畫緣起
### AI模組進行租金預測，並與空間區位決策模型結合  
• 雖近年許多AI人工智慧之技術應用於不同的領域中，然若要將其應用於都市規劃、不動產開發上，受到各項資料數量、尺度有所差異，以及涉及質性資料面向之探討，故實務上以單一AI模組評估土地開發優先級、方式屬困難。  
• 故本研究主要以AI模組進行租金價格預測，並結合傳統規劃領域空間決策考量模式，據以提供整體開發潛力分析。
## 計畫架構
### 自動化空間區位決策模型構想 : 多AI模型並聯架構
• 空間決策為多變因分析過程，仰賴產業、交通、人口、政策、法規、決策者主觀意識等結構性和非結構性知識的全面考量。  
• AI則擅長單一變數趨勢預測，奠基於清楚之問題定義上，和空間決策之問題本質顯然不同。  
• 空間決策的自動化人工智慧模型應為多個AI模型串聯之架構，將空間決策議題劃分為數個子議題，並針對個別議題建立AI模型，再建立基於空間問題的資訊整合機制，將各模型串聯，達成空間決策自動化之目的。  
<div align=center><img width="679.5" height="265.5" src="https://github.com/yichun-hub/RentAI_STGCN/blob/main/graph/1.PNG"/>

## 計畫流程
### AI模組進行租金預測，並與空間區位決策模型結合
<div align=center><img width="522.5" height="337.5" src="https://github.com/yichun-hub/RentAI_STGCN/blob/main/graph/2.PNG"/>

## AI租金預測|AI模組簡介
### 以實價登錄歷史租金資料進行AI建模
• 租金資料建模：  
租金為不動產開發之重要參考指標  
實價登錄平台資料完整、數值化程度高  
實價登錄平台不易進行標的間相互比較  
• AI模型選擇：Spatio-Temporal Graph Convolutional Networks, STGCN 
租金資料圖時具時間維度與空間維度之關聯性  
時空序列資料  
時空圖卷積網絡  
### STGCN：萃取時空特徵之卷積神經網路  
• 模型特色：自動學習資料之Spatial（空間）和Temporal（時間）的特徵  
• 時間卷積模塊 + 空間卷積模塊  
• 訓練資料：民國100~111年桃園市、新北市實價登錄資料  
• 以最小統計區做為單位進行訓練、預測   
<img width="383.5" height="304.5" src="https://github.com/yichun-hub/RentAI_STGCN/blob/main/graph/3.PNG"/>
  
### 區域模型建立：新北、桃園資料獨立建模
<img width="500.5" height="325.5" src="https://github.com/yichun-hub/RentAI_STGCN/blob/main/graph/4.PNG"/>
## AI租金預測|預測成果應用
<img width="489.5" height="294" src="https://github.com/yichun-hub/RentAI_STGCN/blob/main/graph/5.PNG"/>  
• 快速獲得目標土地區位租金行情  
• 作為土地開發區位決策模型參考依據
