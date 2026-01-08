import os
import webbrowser
from flask import Flask, request, jsonify, render_template
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Disable telemetry
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

load_dotenv()
app = Flask(__name__)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.4
)

# --- Tool class ---
class HealthTipTool(BaseTool):
    name: str = "Basic Health Suggestion"
    description: str = "Provides simple mental and physical well-being advice based on the user's message."

    def _run(self, user_input: str) -> str:
        text = user_input.lower()
        if "stress" in text:
            return "Try deep breathing, take short breaks, and get enough sleep."
        elif "sad" in text:
            return "Talk to someone you trust, go for a short walk, and do something you enjoy."
        elif "tired" in text:
            return "Make sure you're hydrated and get proper rest."
        else:
            return "Maintain a balanced routine with proper sleep, hydration, and light exercise."

health_tip_tool = HealthTipTool()

buddy = Agent(
    name="Buddy",
    role="Health Companion",
    goal="Support users emotionally and give basic health advice",
    backstory="Buddy is a friendly AI health companion that supports users with gentle advice.",
    tools=[health_tip_tool],
    llm=llm,
    verbose=False
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "")

        task = Task(
            description=f"User says: {user_input}. Respond kindly and give helpful advice.",
            expected_output="A kind, supportive, and helpful response.",
            agent=buddy
        )

        crew = Crew(
            agents=[buddy],
            tasks=[task],
            process=Process.sequential
        )

        result = crew.kickoff()

        if hasattr(result, "raw"):
            reply = result.raw
        elif hasattr(result, "output"):
            reply = result.output
        else:
            reply = str(result)

        return jsonify({"reply": reply})

    except Exception as e:
        print("❌ ERROR in /chat:", e)
        return jsonify({"reply": f"Backend error: {str(e)}"}), 500

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000")
    app.run(port=5000, debug=True)