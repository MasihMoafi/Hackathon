#!/usr/bin/env python
import os
import sys
import asyncio
import json
import tempfile
import base64
from pathlib import Path
from mcp import ClientSession, stdio_client, StdioServerParameters
from gemma_loader import load_multimodal_system
from agent_system import MultiAgentSystem
import nest_asyncio

nest_asyncio.apply()

def clear_proxy_settings():
    for var in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        if var in os.environ:
            del os.environ[var]

clear_proxy_settings()

class JulietteApp:
    def __init__(self):
        self.gemma_loader = None
        self.agent_system = None
        self.mcp_session = None
        self.project_root = Path(__file__).parent
        
    async def initialize(self):
        print("🔮 Loading Juliette's consciousness...")
        
        self.gemma_loader = load_multimodal_system()
        
        gemini_api_key = os.getenv('GEMINI_API_KEY')
        if gemini_api_key:
            self.gemma_loader.setup_tts(gemini_api_key)
        else:
            print("⚠️ Warning: GEMINI_API_KEY not found. TTS will be disabled.")
            
        self.agent_system = MultiAgentSystem(self.gemma_loader)
        
        host_path = self.project_root / "A-Modular-Kingdom" / "agent" / "host.py"
        if not host_path.exists():
            print(f"❌ MCP host not found at {host_path}")
            return False
            
        print("🔗 Connecting to A-Modular-Kingdom MCP server...")
        params = StdioServerParameters(
            command=sys.executable, 
            args=["-u", str(host_path)]
        )
        
        try:
            self.read, self.write = await stdio_client(params).__aenter__()
            self.mcp_session = await ClientSession(self.read, self.write).__aenter__()
            await self.mcp_session.initialize()
            await self.agent_system.set_mcp_session(self.mcp_session)
            print("✅ MCP connection established!")
            return True
        except Exception as e:
            print(f"❌ MCP connection failed: {e}")
            return False
            
    async def enhanced_agent_response(self, agent_type: str, transcript: str, voice_emotion: str, visual_emotion: str, context: str = "") -> str:
        try:
            search_result = await self.mcp_session.call_tool('search_memories', {'query': transcript, 'top_k': 3})
            memories = json.loads(search_result.content[0].text) if search_result.content else []
            memory_context = "\n".join([f"- {mem.get('content', '')}" for mem in memories if mem and 'content' in mem])
            
            decision_prompt = f"""Decide which external tool to use for this query:
1. "rag" - for document/knowledge questions
2. "web_search" - for current/real-time info
3. "memory" - for custom personal/emotional support

Query: "{transcript}"
Emotion: {voice_emotion}/{visual_emotion}

Respond with JSON: {{"tool": "rag|web_search|none"}}"""

            messages = [{"role": "user", "content": [{"type": "text", "text": decision_prompt}]}]
            inputs = self.gemma_loader.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, 
                return_dict=True, return_tensors="pt"
            ).to("cuda")
            
            outputs = self.gemma_loader.model.generate(
                **inputs, max_new_tokens=64, temperature=0.3, top_p=0.7, top_k=30
            )
            
            decision_response = self.gemma_loader.tokenizer.decode(outputs[0], skip_special_tokens=True)
            tool_decision = decision_response.split("assistant")[-1].strip()
            
            external_context = ""
            try:
                if "rag" in tool_decision.lower():
                    rag_result = await self.mcp_session.call_tool('query_knowledge_base', {'query': transcript})
                    knowledge = json.loads(rag_result.content[0].text)
                    external_context = f"Knowledge: {knowledge.get('result', '')}"
                elif "web_search" in tool_decision.lower():
                    search_result = await self.mcp_session.call_tool('web_search', {'query': transcript})
                    results = json.loads(search_result.content[0].text)
                    external_context = f"Web: {results.get('results', '')}"
            except Exception as e:
                print(f"Tool error: {e}")
            
            agent_config = self.agent_system.router.agents[agent_type]
            enhanced_prompt = f"""{agent_config['prompt']}

MEMORY CONTEXT:
{memory_context}

EXTERNAL KNOWLEDGE:
{external_context}

CONVERSATION HISTORY:
{context}

USER EMOTIONAL STATE:
Voice: {voice_emotion}, Visual: {visual_emotion}

USER INPUT: "{transcript}"

Respond as {agent_config['name']} using all available context. Be empathetic and helpful."""

            messages = [{"role": "user", "content": [{"type": "text", "text": enhanced_prompt}]}]
            inputs = self.gemma_loader.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to("cuda")
            
            outputs = self.gemma_loader.model.generate(
                **inputs, max_new_tokens=256, temperature=1.0, top_p=0.95, top_k=64
            )
            
            response = self.gemma_loader.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.split("assistant")[-1].strip()
            
        except Exception as e:
            print(f"Enhanced response error: {e}")
            return self.agent_system.router.generate_agent_response(agent_type, transcript, voice_emotion, visual_emotion, context)
    
    async def process_input(self, audio_file=None, image_file=None, text_input=None):
        audio_data = None
        image_data = None
        
        if audio_file:
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
                
        if image_file:
            with open(image_file, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()
        
        result = await self.agent_system.process_multimodal_input(
            audio_data=audio_data,
            image_data=image_data,
            transcript=text_input
        )
        
        if self.mcp_session:
            enhanced_response = await self.enhanced_agent_response(
                result['agent_used'],
                result['transcript'],
                result['voice_emotion'],
                result['visual_emotion'],
                self.agent_system.memory.get_recent_context()
            )
            result['response'] = enhanced_response
            
            if result['audio_response']:
                result['response'] = enhanced_response
                result['audio_response'] = self.gemma_loader.generate_tts(enhanced_response)
        
        return result
    
    async def interactive_mode(self):
        print("\n👑 Queen Juliette is ready!")
        print("Commands: 'audio <file>', 'image <file>', 'text <message>', 'exit'")
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() == 'exit':
                    break
                    
                if user_input.startswith('audio '):
                    audio_file = user_input[6:].strip()
                    if os.path.exists(audio_file):
                        result = await self.process_input(audio_file=audio_file)
                        self.display_result(result)
                    else:
                        print(f"❌ Audio file not found: {audio_file}")
                        
                elif user_input.startswith('image '):
                    image_file = user_input[6:].strip()
                    if os.path.exists(image_file):
                        result = await self.process_input(image_file=image_file)
                        self.display_result(result)
                    else:
                        print(f"❌ Image file not found: {image_file}")
                        
                elif user_input.startswith('text '):
                    text_message = user_input[5:].strip()
                    result = await self.process_input(text_input=text_message)
                    self.display_result(result)
                    
                else:
                    result = await self.process_input(text_input=user_input)
                    self.display_result(result)
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def display_result(self, result):
        print(f"\n🎭 Emotions: Voice={result['voice_emotion']}, Visual={result['visual_emotion']}")
        print(f"🤖 Agent: {result['agent_used'].replace('_', ' ').title()}")
        print(f"📝 Transcript: {result['transcript']}")
        print(f"👑 Juliette: {result['response']}")
        
        if result.get('audio_response'):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(base64.b64decode(result['audio_response']))
                print(f"🔊 Audio saved: {f.name}")

async def main():
    app = JulietteApp()
    
    if await app.initialize():
        await app.interactive_mode()
    else:
        print("❌ Initialization failed!")

if __name__ == "__main__":
    asyncio.run(main())