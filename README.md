# Self-Improving RAG Application

A Streamlit application that allows users to chat with PDF documents using advanced RAG (Retrieval-Augmented Generation) techniques. The application uses CrewAI for agent-based interactions and Google's Gemini Pro model for generating responses.

## Features

- PDF document upload
- Interactive chat interface
- Advanced RAG implementation
- Self-improving responses
- Beautiful Streamlit UI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nomanafzal78/self-improving-rag.git
cd self-improving-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run src/app.py
```

## Usage

1. Upload your PDF document using the sidebar
2. Ask questions about the document content
3. Get AI-powered responses based on the document context

## Environment Variables

The application requires the following environment variables:
- `GOOGLE_API_KEY`: Your Google API key for Gemini Pro

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
