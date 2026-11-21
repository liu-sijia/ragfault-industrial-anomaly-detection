# 架构与代码导览

## 端到端流程

1. `src/data_processor.py` 读取 CSV，识别时间、标签与配置的传感器字段，完成启动阶段过滤、基线统计和滑动窗口切分。
2. 每个窗口形成均值、标准差、最大值和 Z-score 等统计描述。
3. `src/knowledge_base.py` 从领域知识文本和规则表构建向量索引；`src/rag_retriever.py` 根据窗口状态检索相关知识。
4. `src/llm_detector.py` 将统计特征和检索上下文组装为 Prompt，通过 Ollama `/api/generate` 调用本地模型，并解析 `Decision` 与 `Reason`。
5. `src/evaluator.py` 计算分类指标；`src/visualizer.py` 生成图表；`src/post_processor.py` 提供预测序列后处理工具。
6. `frontend/server.py` 将流程封装为文件上传、知识库列表、知识库构建和检测 API，静态前端展示结果。

## 场景知识库构建

`src/scene_knowledge_builder.py` 接收工程文档和字段 CSV，使用本地模型抽取传感器说明、异常模式与规则，随后生成与默认知识库相同结构的场景目录。生成内容位于 `knowledge_base/scenes/`，默认不进入版本控制。

## HAI 扩展流程

`extension_experiment/adapters/hai_adapter.py` 将 HAI 格式转换为统一窗口输入；`generic_llm_detector.py` 支持任意字段的特征表；独立 Prompt 和知识文本用于对照有/无 RAG 的流程。配置中的攻击区间是既有实验设置，换用不同数据版本时必须重新核对。

## 设计约束

- 推理依赖本地 Ollama 服务；代码没有云端 API 密钥逻辑。
- FAISS 索引与 Sentence Transformer 模型可能在首次运行时下载/生成。
- LLM 输出具有非确定性，即使温度较低也可能出现边界窗口波动。
- 当前服务绑定 `127.0.0.1`，定位为单机演示，不包含用户认证、队列、数据库或生产级隔离。
- 上传文件会进入临时目录处理；在处理不可信文件前应进一步增加大小、压缩比与资源使用限制。

