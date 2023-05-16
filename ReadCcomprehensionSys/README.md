### 、文件结构
```
 ├── data(存放训练数据和测试数)                
 |  └── ChID   # 多选式任务原始数据集
 |  └── CMRC   # 单跨抽取式任务原始数据集
 ├── model(Bert预训练模型)
 ├── MultipleChoice(多选式任务)
 |  └── data_format   # 数据处理
 |  └── dataset       # 创建dataset
 |  └── main          # 训练主函数
 |  └── predict       # 预测函数
 |  └── trainer       # 训练类
 ├── SingleSpanExtraction(单跨抽取式)
 |  └── BertWithAttention   # 可视化
 |  └── data_format   # 数据处理
 |  └── dataset       # 创建dataset
 |  └── main          # 训练主函数
 |  └── predict       # 预测函数
 |  └── trainer       # 训练类
 ├── templates(模版文件，存储html)
 ├── app.py(运行项目)
```


