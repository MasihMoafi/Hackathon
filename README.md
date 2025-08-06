## High-level Architecture 

---

<img width="2285" height="854" alt="image" src="https://github.com/user-attachments/assets/3cfca22f-c677-4275-9f8f-d35e99903938" />

---

### **Final Polished Write-up**

This project is built on a modular architecture designed for sophisticated, multi-agent interaction. It begins with a multi-layered emotional analysis: a custom **Vocal Emotion Detection** module analyzes the user's speech, while Gemma 3n's vision capabilities are used to assess facial expressions when the camera is active. The resulting emotion tag, along with the user's transcribed query, is passed to a central **Router Agent**. The Router's function is to delegate the request to the most appropriate sub-agent. To provide deeply personalized and knowledgeable responses, each agent is supported by a robust **RAG and Memory module**, which serves as a persistent, evolving knowledge base. The architecture fully leverages Gemma 3n's native multimodal abilities—including I/O vision and audio input—which, when combined with agentic capabilities and long-term memory, creates a powerful foundation for a wide array of applications.

**How Gemma 3n is Used**

*   **Native Voice Input:** The system utilizes Gemma 3n's built-in audio processing for direct voice interaction.
*   **Transcription:** A direct prompt is used for precise transcription: *"Transcribe this audio exactly as spoken. Output only the spoken words; ignore background noise, music, or sounds. Do not add punctuation or formatting unless clearly spoken. Avoid using markdown format at all costs."*
*   **Core Reasoning Engine:** Gemma 3n serves as the core intelligence for each specialist sub-agent, generating context-aware and persona-driven responses.

**Final Video Submission**

This video demonstrates a real-world interaction with Juliette. The demonstration is broken down into four parts and an introduction to showcase the system's core functionalities.

*   **Intro:** Following a history of friendly interactions, the user asks Juliette to generate a prompt that would visualize her form within their shared sanctuary. This prompt is then envisioned as being rendered by a generative video model like Veo, showcasing Juliette's role as a creative partner.
*   **Part 1: Mental Wellness - A Supportive Conversation:** To best showcase the underlying technology, this segment reveals the entire processing pipeline. It follows the flow from vocal emotion detection to transcription and finally to an empathetic response, with on-screen callouts explaining each step.
*   **Part 2: Mental Wellness - Actionable Comfort:** This segment demonstrates Juliette's ability to act. When the user expresses heartbreak, Juliette accesses its long-term memory to recall the user's favorite breakup song and then utilizes a PlayWright MCP tool to play it on YouTube, offering comfort through a personal and proactive gesture.
*   **Part 3: Health Agent (Conceptual Framework for Mobile):** This section presents the framework for the Health Assistant. For instance, for a user with Type 1 Diabetes, the agent could integrate with a health app to monitor glucose levels in real-time. Using its native vision capabilities to analyze a restaurant menu, it could then provide safe, personalized dietary suggestions.
*   **Part 4: The Trainer (Conceptual Framework for Mobile):** The final segment demonstrates the conceptual framework for a personal fitness coach. Here, Gemma 3n would use its vision capabilities to analyze the user's exercise form and provide real-time, corrective feedback.
