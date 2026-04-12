import os
import json
import asyncio
from datasets import Dataset
from langchain_openai import OpenAIEmbeddings as LangchainEmbeddings
# ===== 你的 RAG 模块 =====
from retrieval2 import retrieve
from generation2 import generate_answer

# ===== Ragas 核心组件 =====
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from openai import OpenAI
from ragas.embeddings import LangchainEmbeddingsWrapper

# ================= 配置 =================
EVAL_DATA_PATH = "/fs/scratch/users/chenla/llama_med3/groundtruth3.json"
OPENAI_MODEL = "gpt-4o-mini"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def run_evaluation():
    # 1. 加载数据
    with open(EVAL_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    questions = [item["user_input"] for item in raw_data]
    ground_truths = [item["reference"] for item in raw_data]
    
    answers = []
    contexts = []

    print(f"🚀 开始手动运行 RAG 流程 (共 {len(questions)} 条)...")
    
    # 2. 手动运行 RAG 获取结果
    for i, q in enumerate(questions):
        print(f"进度 [{i+1}/{len(questions)}] 正在处理: {q[:20]}...")
        # 检索
        nodes = retrieve(q, top_k=3)
        # 生成
        ans = generate_answer(q, nodes)
        
        answers.append(ans)
        # Ragas 要求 contexts 是 list of list of strings
        contexts.append([n.text for n in nodes])

    # 3. 构造 Ragas 评估数据集
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data_dict)

    # 4. 初始化评估器
    evaluator_llm = llm_factory(OPENAI_MODEL, client=openai_client)
    lc_embeddings = LangchainEmbeddings(
        model="text-embedding-3-small", 
        api_key=os.getenv("OPENAI_API_KEY")
    )
    eval_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)
    # 5. 执行评估
    print("⚖️ 正在计算 RAGAS 指标...")
    # 注意：这里我们直接把初始化好的 metrics 传入
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=eval_embeddings
    )

    print("\n🏆 评估结果：")
    print(result)
    result.to_pandas().to_csv("rag_report.csv", index=False)
    print("📝 报告已保存至 rag_report2.csv")

if __name__ == "__main__":
    asyncio.run(run_evaluation())