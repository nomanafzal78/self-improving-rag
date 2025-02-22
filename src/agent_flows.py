# agent_flows.py

from crewai import Agent, Task, Crew
import pdfplumber
import os
import google.generativeai as genai
import streamlit as st
import requests
from typing import List, Dict
from datetime import datetime

# Configure the Gemini API
# Using the provided API key
api_key = 'AIzaSyCOjK0ccerOvPCYRx0iO5YEg3fTWA1s7zg'
genai.configure(api_key=api_key)  

class InternetSearchTool:
    def search(self, query: str) -> List[Dict]:
        """Search the internet for information using DuckDuckGo."""
        try:
            # Using DuckDuckGo API for internet search
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_redirect": 1,
                "no_html": 1,
                "skip_disambig": 1,
                "t": "self-improving-rag"  # Custom user agent
            }
            
            st.info("🔍 Sending search request to DuckDuckGo...")
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                st.error(f"Search request failed with status code: {response.status_code}")
                return []
                
            results = response.json()
            
            # Get both RelatedTopics and AbstractText
            topics = results.get('RelatedTopics', [])
            abstract = results.get('AbstractText', '')
            
            # Combine abstract with topics if available
            search_results = []
            if abstract:
                search_results.append({
                    'Text': abstract,
                    'FirstURL': results.get('AbstractURL', '')
                })
            
            # Add topics to results
            search_results.extend(topics[:2])  # Limit to 2 additional results
            
            st.success(f"✅ Found {len(search_results)} relevant results")
            return search_results
            
        except Exception as e:
            st.error(f"Error in internet search: {str(e)}")
            return []

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
        # Enhanced relevance checking
        keywords = query.lower().split()
        text_lower = retrieved_info.lower()
        
        # Check if any keyword is present in the text
        keyword_present = any(keyword in text_lower for keyword in keywords)
        
        # Check if the content seems relevant to the query
        relevance_score = sum(1 for keyword in keywords if keyword in text_lower) / len(keywords)
        
        is_relevant = keyword_present and relevance_score > 0.3
        
        if not is_relevant:
            st.warning("⚠️ Content in PDF appears to be not relevant to the query")
        
        return is_relevant

# Define the internet search agent
class InternetSearchAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Internet Researcher",
            goal="Find relevant information from the internet when PDF content is insufficient",
            backstory="I am an expert at searching the internet for accurate and relevant information."
        )
        self.search_tool = InternetSearchTool()
        
    def run(self, query):
        st.info("🌐 Searching the internet for information...")
        search_results = self.search_tool.search(query)
        
        if search_results:
            # Combine search results into a single text
            combined_info = "\n\n".join([
                f"Source {i+1}:\n{result.get('Text', '')}\nURL: {result.get('FirstURL', '')}"
                for i, result in enumerate(search_results) if result.get('Text')
            ])
            
            if combined_info.strip():
                st.success("✅ Found relevant information from the internet")
                return combined_info
                
        st.warning("⚠️ No relevant information found on the internet")
        return ""

# Define the generation agent
class GenerationAgent(Agent):
    def __init__(self):
        super().__init__(
            role="Response Generator",
            goal="Generate clear and accurate responses based on all available information",
            backstory="I am an expert at crafting informative responses using the Gemini model, ensuring accuracy and clarity."
        )
        
    def run(self, relevant_info, query, internet_info=""):
        model = genai.GenerativeModel('gemini-pro')
        
        # Combine PDF and internet information if available
        if internet_info:
            prompt = f"""Based on the following information, please answer this question: {query}

PDF Content: {relevant_info}

Internet Search Results:
{internet_info}

Please provide a comprehensive answer that:
1. Addresses the question directly
2. Combines information from both sources when relevant
3. Cites internet sources when used
4. Maintains accuracy and clarity"""
        else:
            prompt = f"""Based on the following content, please answer this question: {query}

Content: {relevant_info}

Please provide a clear and accurate answer based on the provided content."""
        
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "Sorry, I encountered an error while generating the response."

def test_agents(query):
    # Initialize agents
    retrieval_agent = RetrievalAgent()
    conditional_agent = ConditionalAgent()
    internet_agent = InternetSearchAgent()
    generation_agent = GenerationAgent()

    try:
        # Step 1: Retrieve information from PDF
        st.info("📄 Retrieving information from PDF...")
        retrieved_info = retrieval_agent.run(query)

        # Step 2: Check relevance
        st.info("🔍 Analyzing content relevance...")
        is_relevant = conditional_agent.run(retrieved_info, query)

        # Step 3: If not relevant or no clear answer, search internet
        internet_info = ""
        if not is_relevant:
            internet_info = internet_agent.run(query)

        # Step 4: Generate response
        st.info("💭 Generating response...")
        
        if internet_info:
            response = generation_agent.run(retrieved_info, query, internet_info)
            return f"Combined Response (including internet search results):\n\n{response}"
        elif is_relevant:
            response = generation_agent.run(retrieved_info, query)
            return response
        else:
            return "I could not find relevant information in either the PDF or internet search results. Please try rephrasing your question or provide more context."
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return f"Error processing your request: {str(e)}" 