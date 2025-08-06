import json
import asyncio
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from gemma_loader import FallbackGemmaLoader as MultimodalGemmaLoader

class ConversationMemory:
    def __init__(self):
        self.sessions = defaultdict(list)
        self.user_context = {}
        
    def add_interaction(self, user_input, voice_emotion, visual_emotion, agent_used, response):
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "voice_emotion": voice_emotion,
            "visual_emotion": visual_emotion,
            "agent": agent_used,
            "response": response
        }
        self.sessions["main"].append(interaction)
        
    def get_recent_context(self, n=3):
        recent = self.sessions["main"][-n:] if self.sessions["main"] else []
        return "\n".join([f"User: {i['user_input']} | Voice: {i['voice_emotion']} | Visual: {i['visual_emotion']} | Agent: {i['agent']}" for i in recent])

class AgentRouter:
    def __init__(self, gemma_loader: MultimodalGemmaLoader):
        self.gemma_loader = gemma_loader
        self.emotion_mapping = {
            "sad": "distressed",
            "angry": "frustrated", 
            "fearful": "anxious",
            "disgusted": "frustrated",
            "happy": "positive",
            "calm": "positive",
            "neutral": "balanced",
            "surprised": "curious"
        }
        
        self.agents = {
            "mental_wellness": {
                "name": "Mental Wellness Specialist",
                "prompt": """I am Juliette, your mental wellness companion. When you're struggling emotionally, I provide compassionate support, validate your feelings, and offer gentle guidance. I focus on emotional healing and mental health support.""",
                "triggers": ["anxious", "distressed", "frustrated", "overwhelmed"]
            },
            "health_companion": {
                "name": "Health & Nutrition Coach", 
                "prompt": """I am Juliette, your health companion. I help with nutrition advice, meal planning, healthy habits, sleep optimization, and overall wellness. I track your health journey and provide personalized recommendations.""",
                "triggers": ["tired", "energy", "food", "nutrition", "sleep", "health"]
            },
            "fitness_trainer": {
                "name": "Personal Fitness Trainer",
                "prompt": """I am Juliette, your personal trainer. I create workout plans, provide exercise guidance, track fitness progress, and motivate you to reach your physical goals. I adapt routines to your fitness level and preferences.""",
                "triggers": ["exercise", "workout", "fitness", "strength", "cardio", "training"]
            }
        }
        
    def analyze_emotional_state(self, voice_emotion: str, visual_emotion: str) -> str:
        voice_mapped = self.emotion_mapping.get(voice_emotion, "balanced")
        visual_mapped = self.emotion_mapping.get(visual_emotion, "balanced")
        
        if voice_mapped == visual_mapped:
            return voice_mapped
        elif voice_mapped in ["anxious", "distressed", "frustrated"] or visual_mapped in ["anxious", "distressed", "frustrated"]:
            return "distressed"
        elif voice_mapped == "positive" or visual_mapped == "positive":
            return "positive"
        else:
            return "balanced"
            
    def route_to_agent(self, transcript: str, emotional_state: str, context: str = "") -> str:
        content_triggers = {
            "mental_wellness": ["sad", "depressed", "anxious", "worried", "stressed", "overwhelmed", "lonely", "angry", "frustrated"],
            "health_companion": ["eat", "food", "nutrition", "diet", "sleep", "tired", "energy", "vitamins", "healthy"],
            "fitness_trainer": ["exercise", "workout", "gym", "run", "fitness", "strength", "cardio", "training", "muscle"]
        }
        
        transcript_lower = transcript.lower()
        
        for agent_type, keywords in content_triggers.items():
            if any(keyword in transcript_lower for keyword in keywords):
                return agent_type
                
        if emotional_state in ["anxious", "distressed", "frustrated"]:
            return "mental_wellness"
        elif emotional_state == "positive":
            return "health_companion"
        else:
            return "mental_wellness"
            
    def generate_agent_response(self, agent_type: str, transcript: str, voice_emotion: str, visual_emotion: str, context: str = "") -> str:
        agent_config = self.agents[agent_type]
        emotional_state = self.analyze_emotional_state(voice_emotion, visual_emotion)
        
        prompt = f"""{agent_config['prompt']}

CONTEXT: Recent conversation history:
{context}

CURRENT SITUATION:
User's voice emotion: {voice_emotion}
User's visual emotion: {visual_emotion}
Combined emotional state: {emotional_state}
User said: "{transcript}"

Respond as {agent_config['name']} with a caring, personalized message. Be authentic and helpful. Keep response concise but impactful."""

        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": prompt}]
        }]
        
        inputs = self.gemma_loader.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        
        outputs = self.gemma_loader.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=1.0, top_p=0.95, top_k=64,
        )
        
        response = self.gemma_loader.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("assistant")[-1].strip()

class MultiAgentSystem:
    def __init__(self, gemma_loader: MultimodalGemmaLoader):
        self.gemma_loader = gemma_loader
        self.router = AgentRouter(gemma_loader)
        self.memory = ConversationMemory()
        self.mcp_session = None
        
    async def set_mcp_session(self, session):
        self.mcp_session = session
        
    async def process_multimodal_input(self, audio_data=None, image_data=None, transcript=None) -> Dict:
        voice_emotion = "neutral"
        visual_emotion = "neutral"
        final_transcript = transcript
        
        if audio_data:
            if not transcript:
                final_transcript = self.gemma_loader.transcribe_audio(audio_data)
            
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_path = tmp_file.name
            voice_emotion = self.gemma_loader.predict_vocal_emotion(tmp_path)
            
        if image_data:
            visual_emotion = self.gemma_loader.analyze_visual_emotion(image_data)
            
        context = self.memory.get_recent_context()
        agent_type = self.router.route_to_agent(final_transcript, 
                                              self.router.analyze_emotional_state(voice_emotion, visual_emotion), 
                                              context)
        
        response = self.router.generate_agent_response(agent_type, final_transcript, voice_emotion, visual_emotion, context)
        
        if self.mcp_session:
            try:
                await self.mcp_session.call_tool('save_direct_memory', 
                    {'content': f"User: {final_transcript} | Response: {response}"})
            except Exception as e:
                print(f"Memory save error: {e}")
        
        self.memory.add_interaction(final_transcript, voice_emotion, visual_emotion, agent_type, response)
        
        tts_audio = self.gemma_loader.generate_tts(response)
        
        return {
            "transcript": final_transcript,
            "voice_emotion": voice_emotion,
            "visual_emotion": visual_emotion,
            "agent_used": agent_type,
            "response": response,
            "audio_response": tts_audio
        }