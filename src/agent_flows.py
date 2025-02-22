# agent_flows.py

from crewai import Agent, Task, Crew
import pdfplumber
import os
import google.generativeai as genai
import streamlit as st

# Configure the Gemini API
# Using the provided API key
api_key = 'AIzaSyCOjK0ccerOvPCYRx0iO5YEg3fTWA1s7zg'
genai.configure(api_key=api_key)  

# Define the retrieval agent
class RetrievalAgent(Agent):
    def __init__(self):
        super().__init__(
            role="PDF Content Retriever",
            goal="Extract and provide relevant information from PDF documents",
            backstory="I am an expert at reading and extracting information from PDF documents. I ensure all content is properly retrieved and formatted."
        )
        
    def run(self, query):
        # Use the PDF path from session state
        if st.session_state.pdf_path:
            with pdfplumber.open(st.session_state.pdf_path) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() or ''
            return text
        else:
            raise Exception("No PDF file uploaded")

# Define the conditional agent
class ConditionalAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Relevance Analyzer",
            goal="Determine if retrieved information is relevant to the user's query",
            backstory="I am specialized in analyzing content relevance and ensuring responses match user queries accurately."
        )
        
    def run(self, retrieved_info, query):
        keywords = query.lower().split()
        text_lower = retrieved_info.lower()
        return any(keyword in text_lower for keyword in keywords)

# Define the generation agent
class GenerationAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Response Generator",
            goal="Generate clear and accurate responses based on relevant information",
            backstory="I am an expert at crafting informative responses using the Gemini model, ensuring accuracy and clarity."
        )
        
    def run(self, relevant_info, query):
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"Based on the following content, please answer this question: {query}\n\nContent: {relevant_info}"
        response = model.generate_content(prompt)
        return response.text

def test_agents(query):
    # Initialize agents
    retrieval_agent = RetrievalAgent()
    conditional_agent = ConditionalAgent()
    generation_agent = GenerationAgent()

    # Step 1: Retrieve information
    retrieved_info = retrieval_agent.run(query)

    # Step 2: Check relevance
    is_relevant = conditional_agent.run(retrieved_info, query)

    # Step 3: Generate response if relevant
    if is_relevant:
        response = generation_agent.run(retrieved_info, query)
        return response
    else:
        return "No relevant information found in the document." 