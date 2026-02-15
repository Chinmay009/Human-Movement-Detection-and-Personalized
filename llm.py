from ollama import Ollama

def generate_advisory(user_info, activity_info, model_name="llama3"):
    prompt = f"""
You are a concise health advisory assistant.
User: Age: {user_info['Age']} Gender: {user_info['Gender']} Weight: {user_info['Weight']} Location: {user_info['Location']} Goal: {user_info['Goal']}
Current State: Activity: {activity_info['Activity']} Duration: {activity_info['Duration']} minutes Temperature: {activity_info['Temperature']}°C
Rules:
- No medical diagnosis
- No medication advice
- Keep it short
- Max 3 bullet points
- Each bullet under 20 words
- Only mention relevant suggestions
Output format:
• Observation
• Safety note
• Suggested adjustment (if needed)
"""
    ollama = Ollama(model=model_name)
    return ollama.chat(prompt)
