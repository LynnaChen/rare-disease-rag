from config import (
    get_generation_model, 
    GENERATION_SYSTEM_TEMPLATE, 
    GENERATION_USER_TEMPLATE,
    REJECTION_RESPONSE
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole

def build_context_from_parents(parent_nodes, max_chars=6000):
    """
    从检索到的节点中构建上下文。
    包含了文件名和页码的提取，以便 LLM 能够进行引用。
    """
    context_parts = []
    total_len = 0

    for i, node in enumerate(parent_nodes):
        text = node.text.strip()
        if not text:
            continue

        # 提取元数据：文件名和页码
        file_name = node.metadata.get('file_name', 'Unknown File')
        page_no = node.metadata.get('page_label', 'N/A')

        # 构造带编号和来源的片段
        part = f"[Document {i+1}: {file_name}, Page {page_no}]\n{text}\n"
        
        if total_len + len(part) > max_chars:
            break

        context_parts.append(part)
        total_len += len(part)

    return "\n\n".join(context_parts)


def generate_answer(query: str, parent_nodes):
    """
    基于检索到的内容，调用配置好的 LLM 生成回答。
    """
    # 1. 检查是否有参考资料，如果没有，直接返回 config 里的拒答话术
    if not parent_nodes:
        return REJECTION_RESPONSE

    # 2. 获取 LLM 实例
    llm = get_generation_model()

    # 3. 构造上下文
    context_str = build_context_from_parents(parent_nodes)

    # 4. 【关键修正】：使用 config.py 中的模板填充内容
    # 这样你在 config 里改提示词，这里才会生效
    system_content = GENERATION_SYSTEM_TEMPLATE.format(context=context_str)
    user_content = GENERATION_USER_TEMPLATE.format(question=query)

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_content),
        ChatMessage(role=MessageRole.USER, content=user_content),
    ]

    # 5. 调用模型生成
    response = llm.chat(messages)

    return response.message.content.strip()
