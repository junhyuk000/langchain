from flask import Flask, request, render_template, session
import json
from logic import build_and_visualize_graph

app = Flask(__name__)
app.secret_key = "secret"  # 세션 관리용

graph = build_and_visualize_graph()

@app.route("/", methods=["GET", "POST"])
def index():
    if "history" not in session:
        session["history"] = []

    response = None
    if request.method == "POST":
        user_input = request.form["message"]
        print(f"📨 유저 입력: {user_input}")  # <-- 이거 넣어서 로그 찍어봐!
        state = {"input": user_input}  # ✅ 그대로 사용

        print(f"📦 전달된 state: {state}")  # ✅ 이거 추가해봐
        result = graph.invoke(state)

        # 방어 코드 추가
        if not result or "response" not in result:
            output = "⚠️ 처리 중 오류가 발생했습니다."
        else:
            output = result.get("response", "⚠️ 응답이 없습니다.")


        session["history"].append({"input": user_input, "output": output})
        session.modified = True
        response = output

        # 로그 파일 저장
        with open("chat_log.json", "w", encoding="utf-8") as f:
            json.dump(session["history"], f, indent=2, ensure_ascii=False)

    return render_template("index.html", history=session["history"], response=response)

if __name__ == "__main__":
    app.run(debug=True)
