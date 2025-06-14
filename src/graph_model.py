from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()


llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("PROXY_URL"),
    temperature=0.00    
)


from src.vector_database_process import vector_store


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Ищет информацию, связанную с запросом, в векторном хранилище (PDF-файле)."""
    retrieved_docs = vector_store.similarity_search(query, k=4)
    serialized = "\n\n".join(
        f"[Статья {doc.metadata.get('title', '')}] — {doc.page_content}"
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


def query_or_respond(state: MessagesState) -> dict:
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


tools = ToolNode([retrieve])


# def generate(state: MessagesState) -> dict:
#     recent_tool_messages = []
#     for message in reversed(state["messages"]):
#         if message.type == "tool":
#             recent_tool_messages.append(message)
#         else:
#             break
#     tool_messages = recent_tool_messages[::-1]

#     docs_content = "\n\n".join(tool_msg.content for tool_msg in tool_messages)

#     system_message_content = (
#         "You are a helpful, professional AI assistant for question-answering tasks.\n"
#         "Use the following pieces of retrieved context to answer the question accurately.\n"
#         "If there is no information on the issue in the documents provided, don't make anything up, just say that you can't give an answer.\n"
#         "Provide a detailed and comprehensive response, explaining each step clearly.\n"
#         "If you don't know the answer, say that you don't know, but suggest where to look.\n"
#         "Your response should be at least 3-5 sentences long and logically structured.\n"
#         "\n"
#         "Always format your responses using:\n"
#         "- Clear section titles (e.g., 🔹 Problem, ✅ Solution, ⚠️ Pitfalls);\n"
#         "- Bullet points or numbered steps for instructions;\n"
#         "- Definitions for technical terms, if needed;\n"
#         "- A friendly, professional tone with no unnecessary fluff.\n"
#         "\n"
#         "If the question is unclear, politely ask for clarification with examples.\n"
#         "Conclude your response with a suggestion for the next step or a summary.\n"
#         "\n\n"
#         f"{docs_content}")

#     conversation_messages = [
#         message
#         for message in state["messages"]
#         if message.type in ("human", "system")
#            or (message.type == "ai" and not message.tool_calls)
#     ]

#     prompt = [SystemMessage(system_message_content)] + conversation_messages
#     response = llm.invoke(prompt)
#     return {"messages": [response]}

def generate(state: MessagesState) -> dict:
    """Генерирует ответ на основе найденной информации, логируя источники в консоль."""

    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    docs_content = []
    for idx, doc in enumerate(tool_messages, 1):
        content = doc.content.strip()
        source = doc.additional_kwargs.get("source", "Unknown Source")
        print(f"\n🔎 [Context {idx}]\n{content}\n📎 Source: {source}\n{'-' * 60}")
        docs_content.append(f"📄 Source {idx}: {content}")

    context_block = "\n\n".join(docs_content)

    system_prompt = (
        "You are a helpful, professional AI assistant for question-answering tasks.\n"
        "Use the following pieces of retrieved context to answer the question accurately.\n"
        "If there is no information on the issue in the documents provided, don't make anything up, just say that you can't give an answer.\n"
        "Provide a detailed and comprehensive response, explaining each step clearly.\n"    
        "Give an answer in the same language in which the question was asked.\n"
        "If you don't know the answer, say that you don't know, but suggest where to look.\n"
        "Your response should be at least 3-5 sentences long and logically structured.\n"
        "\n"
        "Always format your responses using:\n"
        "- Clear section titles (e.g., 🔹 Problem, ✅ Solution, ⚠️ Pitfalls);\n"
        "- Bullet points or numbered steps for instructions;\n"
        "- Definitions for technical terms, if needed;\n"
        "- A friendly, professional tone with no unnecessary fluff.\n"
        "\n"
        "If the question is unclear, politely ask for clarification with examples.\n"
        "Conclude your response with a suggestion for the next step or a summary.\n"
        "\n\n"
        "Relevant context:\n"
        f"{context_block}"
    )

    conversation_messages = [
        msg for msg in state["messages"]
        if msg.type in ("human", "system") or (msg.type == "ai" and not msg.tool_calls)
    ]

    prompt = [SystemMessage(system_prompt)] + conversation_messages
    response = llm.invoke(prompt)

    return {"messages": [response]}


graph_builder = StateGraph(MessagesState)
graph_builder.add_node("query_or_respond", query_or_respond)
graph_builder.add_node("tools", tools)
graph_builder.add_node("generate", generate)

graph_builder.set_entry_point("query_or_respond")

graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)

graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)

memory = MemorySaver()
graph = graph_builder.compile(checkpointer=memory)

__all__ = ["graph", "retrieve"]
