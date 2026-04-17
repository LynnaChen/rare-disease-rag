from config import (
    get_generation_model, 
    GENERATION_SYSTEM_TEMPLATE, 
    GENERATION_USER_TEMPLATE,
    REJECTION_RESPONSE
)
from llama_index.core.base.llms.types import ChatMessage, MessageRole

def build_context_from_parents(parent_nodes, max_chars=6000):
    """
    Build context from retrieved parent nodes.
    Includes file name and page information so the LLM can cite sources.
    """
    context_parts = []
    total_len = 0

    for i, node in enumerate(parent_nodes):
        text = node.text.strip()
        if not text:
            continue

        # Extract metadata: file name and page label
        file_name = node.metadata.get('file_name', 'Unknown File')
        page_no = node.metadata.get('page_label', 'N/A')

        # Build a numbered, sourced snippet
        part = f"[Document {i+1}: {file_name}, Page {page_no}]\n{text}\n"
        
        if total_len + len(part) > max_chars:
            break

        context_parts.append(part)
        total_len += len(part)

    return "\n\n".join(context_parts)


def generate_answer(query: str, parent_nodes):
    """
    Generate an answer using the configured LLM based on retrieved content.
    """
    # 1. If there are no sources, return the rejection response from config.
    if not parent_nodes:
        return REJECTION_RESPONSE

    # 2. Get the LLM instance
    llm = get_generation_model()

    # 3. Build context
    context_str = build_context_from_parents(parent_nodes)

    # 4. Key detail: fill templates from config.py so edits there take effect here.
    system_content = GENERATION_SYSTEM_TEMPLATE.format(context=context_str)
    user_content = GENERATION_USER_TEMPLATE.format(question=query)

    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_content),
        ChatMessage(role=MessageRole.USER, content=user_content),
    ]

    # 5. Call the model
    response = llm.chat(messages)

    return response.message.content.strip()
