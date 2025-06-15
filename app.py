import streamlit as st
import json
from dataclasses import dataclass
from typing import List, Dict, Any
import time
from ai_backend.chat import chat_llm, chat
from ai_backend.chat_pdf import chat_llm_pdf

@dataclass
class ChatMessage:
    user_query: str
    transformed_query: str
    context: str
    answer: str
    latency: float = 0.0  
    token_count: int = 0
    similarity_scores: List[float] = None

def invoke_chat_agent(agent_type: str, user_query: str) -> Dict[str, Any]:
    """Invoke the selected chat agent and measure latency"""
    start_time = time.time()
    
    if agent_type == "Daily Mail Reporter":
        response = chat_llm.invoke(chat(user_query=user_query, transformed_query="", context="", answer=""))
    else:  # PDF Chat
        response = chat_llm_pdf.invoke(chat(user_query=user_query, transformed_query="", context="", answer=""))
    
    end_time = time.time()
    latency = end_time - start_time
    
    # Add latency to the response
    response['latency'] = latency
    
    return response

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent_selected' not in st.session_state:
        st.session_state.agent_selected = None

def display_chat_message(message: Dict[str, Any], message_index: int):
    """Display a chat message with expandable sections"""
    with st.container():
        # User message
        st.markdown(f"**You:** {message['user_query']}")
        
        # AI response
        st.markdown(f"**AI:** {message['answer']}")
        
        # Expandable sections for context, transformed query, latency, token count, and similarity scores
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            with st.expander("🔍 View Context", expanded=False):
                if message['context']:
                    st.text_area(
                        "Context:",
                        value=message['context'],
                        height=200,
                        key=f"context_{message_index}",
                        disabled=True
                    )
                else:
                    st.write("No context available")
        
        with col2:
            with st.expander("🔄 View Transformed Query", expanded=False):
                if message['transformed_query']:
                    st.text_area(
                        "Transformed Query:",
                        value=message['transformed_query'],
                        height=100,
                        key=f"transformed_{message_index}",
                        disabled=True
                    )
                else:
                    st.write("No transformed query available")
        
        with col3:
            with st.expander("⏱️ View Latency", expanded=False):
                latency = message.get('latency', 0.0)
                st.markdown("**Latency for query transformation, vector search and LLM response**")
                st.markdown(f"**Time taken:** {latency:.2f} seconds")
                st.markdown(f"**Time taken:** {latency*1000:.0f} milliseconds")
        
        with col4:
            with st.expander("🔢 View Token Count", expanded=False):
                token_count = message.get('token_count', 0)
                st.markdown("**Token Usage**")
                st.markdown(f"**Total tokens:** {token_count}")
                if token_count > 0:
                    st.markdown(f"**Estimated cost:** ~${token_count * 0.00002:.6f}")
        
        with col5:
            with st.expander("📊 View Similarity Scores", expanded=False):
                similarity_scores = message.get('similarity_scores', [])
                st.markdown("**Similarity scores for k=2 documents retrieved**")
                if similarity_scores:
                    for i, score in enumerate(similarity_scores, 1):
                        st.markdown(f"**Document {i}:** {score:.4f}")
                else:
                    st.write("No similarity scores available")
        
        st.divider()

def main():
    st.set_page_config(
        page_title="DocuSummarize AI",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Header
    st.title("🤖 Genesys Chat")
    st.markdown("Choose your chat agent and start summarizing!")
    
    # Sidebar for agent selection
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Agent selection
        agent_type = st.selectbox(
            "Select Chat Agent:",
            ["Daily Mail Reporter", "FAST Anual Report Agent"],
            index=0 if st.session_state.agent_selected is None else (
                0 if st.session_state.agent_selected == "Daily Mail Reporter" else 1
            )
        )
        
        # Update session state if agent changed
        if st.session_state.agent_selected != agent_type:
            st.session_state.agent_selected = agent_type
            st.rerun()
        
        st.markdown("---")
        
        # Agent info
        if agent_type == "Daily Mail Reporter":
            st.info("📝 **Daily Mail Reporter Agent**\n\nAnswers questions and summarizes news articles from the CNN DailyMail dataset.")
        else:
            st.info("📄 **FAST Anual Report Agent**\n\nExtracts insights and summaries from a PDF-based FAST Annual Report.")
        
        st.markdown("---")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.messages = []
            st.rerun()
        
        # Chat statistics
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")
    
    # Main chat interface
    if st.session_state.agent_selected:
        st.markdown(f"### 💬 Chat with {st.session_state.agent_selected}")
        
        # Chat history container
        chat_container = st.container()
        
        with chat_container:
            if st.session_state.messages:
                st.markdown("#### Chat History")
                for i, message in enumerate(st.session_state.messages):
                    display_chat_message(message, i)
            else:
                st.markdown("*No messages yet. Start a conversation below!*")
        
        # Chat input
        st.markdown("#### Send a Message")
        
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Your message:",
                height=100,
                placeholder="Type your message here...",
                key="user_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                submit_button = st.form_submit_button("Send 📤", type="primary")
            with col2:
                if st.form_submit_button("Clear Input 🗑️"):
                    st.rerun()
        
        # Process user input
        if submit_button and user_input.strip():
            with st.spinner(f"Processing with {st.session_state.agent_selected}..."):
                try:
                    # Get response from selected agent
                    response = invoke_chat_agent(st.session_state.agent_selected, user_input.strip())
                    
                    # Add to chat history
                    st.session_state.messages.append(response)
                    
                    # Rerun to display new message
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.error("Please check your backend connections and try again.")
    
    else:
        # Initial selection screen
        st.markdown("### 🚀 Get Started")
        st.markdown("Please select a chat agent from the sidebar to begin chatting.")
        
        # Agent descriptions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            #### 📝 Daily Mail Reporter Agent
            - General purpose conversations
            - Wide range of topics
            - Standard AI responses
            """)
        
        with col2:
            st.markdown("""
            #### 📄 FAST Anual Report Agent
            - PDF document analysis
            - Document-based queries
            - Specialized for file content
            """)

if __name__ == "__main__":
    main()
    


