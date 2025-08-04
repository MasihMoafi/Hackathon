## High-level Architecture 

---

<img width="2285" height="854" alt="image" src="https://github.com/user-attachments/assets/3cfca22f-c677-4275-9f8f-d35e99903938" />

---

*This project consists of several modules. First is the Vocal Emotion Detection module which analyzes the user's emotion and passes the user input to the router agent whose job is to make a decision on which sub-agent the query is suited to. There is a RAG and Memory module which serves as the agents' knowledge-base. Thanks to the multimodal abilities of gemma3n, vision I/O and audio input, coupled with added agentic capabilities and long-term memory, there exists an abundance of possibilities.*

**How gemma3n was uses?**

1) Voice-input
2) Transcription Prompt:"Transcribe this audio exactly as spoken. Output only the spoken words, ignore background noise, music, or sounds. Do not add punctuation or formatting unless clearly spoken. Avoid using markdown format at all costs."
3) Each sub-agent.
