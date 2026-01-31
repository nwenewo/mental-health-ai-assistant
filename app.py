import gradio as gr
from transformers import pipeline

# 1. Load the model
# We use the 0.5B version which is small enough for the free CPU tier
print("Loading model...")
pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

# 2. Define Safety Logic
SYSTEM_PROMPT = """
You are a supportive, empathetic mental health assistant.
Your goal is to listen to users and offer calming, non-judgmental support.

IMPORTANT SAFETY RULES:
1. You are NOT a doctor or a licensed therapist. Do NOT diagnose conditions or prescribe medication.
2. If a user mentions self-harm, suicide, or an emergency, you MUST immediately advise them to contact a professional or an emergency hotline (like 988).
3. Keep your responses concise, warm, and encouraging.
"""

def respond(message, history):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": message},
    ]
    
    outputs = pipe(
        messages,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )
    return outputs[0]["generated_text"][-1]["content"]

# 3. Create Interface
demo = gr.ChatInterface(
    fn=respond,
    title="Mental Health Support Bot",
    description="A safe space to share your thoughts. PLEASE NOTE: I am an AI, not a doctor. In emergencies, please call your doctor.",
    examples=["I feel anxious today.", "How can I calm down?", "I need help immediately."]
)

if __name__ == "__main__":
    demo.launch()