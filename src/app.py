import streamlit as st
import os
from agent_flows import test_agents
import tempfile

# Configure page
st.set_page_config(
    page_title="Self-Improving RAG Application",
    page_icon="📚",
    layout="wide"
)

# Initialize session state for storing the PDF path
if 'pdf_path' not in st.session_state:
    st.session_state.pdf_path = None

# Title and description
st.title('📚 Self-Improving RAG Application')
st.markdown("""
This application allows you to upload a PDF and ask questions about its content.
The AI will analyze the document and provide relevant answers using advanced RAG techniques.
""")

# Create a sidebar for PDF upload
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload your PDF", type=['pdf'])
    
    if uploaded_file:
        # Create a temporary file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "uploaded_pdf.pdf")
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.session_state.pdf_path = temp_path
        st.success("✅ PDF uploaded successfully!")

# Main content area
if st.session_state.pdf_path:
    st.header("Chat with your PDF")
    
    # Query input
    query = st.text_input("What would you like to know about the document?")
    
    if st.button('Ask Question', key='ask'):
        if query:
            with st.spinner('Processing your question...'):
                try:
                    # Display the steps
                    progress_text = st.empty()
                    
                    progress_text.text("🔍 Retrieving information from PDF...")
                    response = test_agents(query)
                    
                    # Create a nice box for the response
                    st.markdown("### Answer")
                    st.markdown(f"```{response}```")
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question.")
else:
    # Display welcome message when no PDF is uploaded
    st.markdown("""
    ### 👋 Welcome!
    
    To get started:
    1. Upload a PDF document using the sidebar
    2. Ask questions about the content
    3. Get AI-powered responses
    
    The system will analyze your document and provide relevant answers to your questions.
    """) 