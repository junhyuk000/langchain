from transformers import pipeline
from langgraph.graph import StateGraph, END
import matplotlib.pyplot as plt
import networkx as nx
import os

# 모델 로드
classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")



# 감정 분류 노드
def classify_emotion(state: dict):
    print(f"🧾 받은 state: {state}")  # ✅ 이것도 찍어보자
    text = state.get("input", "")
    print(f"📩 입력 받은 문장: {text}")  # ✅ 디버깅용

    if not text.strip():
        state["response"] = "⚠️ 입력이 비어 있습니다."
        state["emotion"] = "error"
        print("⚠️ 입력이 공백이므로 error 처리")
        return state

    result = classifier(text)
    print(f"🔍 모델 결과: {result}")

    if not result:
        state["response"] = "⚠️ 감정 분석 실패"
        state["emotion"] = "error"
        return state

    score = int(result[0]["label"][0])
    state["emotion"] = "positive" if score >= 4 else "negative"
    print(f"✅ 분류된 감정: {state['emotion']}")
    return state


# 긍정 응답 노드
def handle_positive(state: dict):
    state["response"] = "🟢 긍정적인 감정이네요! 좋은 하루 보내세요 :)"
    return state

# 부정 응답 노드
def handle_negative(state: dict):
    state["response"] = "🔴 힘들 수 있지만, 잘 해낼 수 있어요! 파이팅!"
    return state

# 수동 시각화 함수
def visualize_graph_manual(sg, filename=None):
    # Flask 기준 static 경로
    base_dir = os.path.dirname(os.path.abspath(__file__))  # logic.py 경로
    static_path = os.path.join(base_dir, "static")
    os.makedirs(static_path, exist_ok=True)

    # graph.png 경로
    if filename is None:
        filename = os.path.join(static_path, "graph.png")

    print(f"📁 그래프 저장 경로: {filename}")

    G = nx.DiGraph()
    G.add_node("classify")
    G.add_node("positive")
    G.add_node("negative")
    G.add_node("END")

    G.add_edge("classify", "positive", label="emotion=positive")
    G.add_edge("classify", "negative", label="emotion=negative")
    G.add_edge("positive", "END")
    G.add_edge("negative", "END")

    pos = nx.spring_layout(G)
    plt.figure(figsize=(5, 5))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'))
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print("✅ 그래프 이미지 저장 완료")


# 그래프 생성 + 시각화 + 컴파일
def build_and_visualize_graph():
    sg = StateGraph(dict)
    sg.add_node("classify", classify_emotion)
    sg.add_node("positive", handle_positive)
    sg.add_node("negative", handle_negative)

    sg.set_entry_point("classify")
    sg.add_conditional_edges("classify", lambda s: s.get("emotion", "error"), {
        "positive": "positive",
        "negative": "negative",
        "error": "negative"
    })
    sg.add_edge("positive", END)
    sg.add_edge("negative", END)

    # ✅ 이거 빠졌으면 절대 graph.png 안 생김
    print("📌 시각화 함수 호출 시작")  # 추가
    visualize_graph_manual(sg)
    print("✅ 시각화 함수 호출 완료")  # 추가

    return sg.compile()
