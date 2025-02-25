# SAP Installation Search Automation Using Agentic AI and Multi-Modal RAG

## Overview

The "SAP Installation Search Automation Using Agentic AI and Multi-Modal RAG" project aims to streamline the process of searching for SAP installation documentation and resources using advanced AI techniques. By leveraging Agentic AI and Multi-Modal Retrieval-Augmented Generation (RAG), this project enhances the efficiency and accuracy of information retrieval, making it easier for users to find relevant installation guides, troubleshooting tips, and best practices.

In today's fast-paced enterprise environments, keeping up with the vast amount of technical documentation can be challenging. This project addresses these challenges by automating the search process, thereby reducing manual efforts and ensuring that critical information is accessible quickly. The integration of cutting-edge AI allows the system to intelligently parse through varied data sources and deliver comprehensive, context-aware results.

Moreover, the project is designed to evolve with user needs and technological advancements. By combining multi-modal retrieval techniques with robust document embedding strategies, it not only identifies relevant information from text but also processes images and tables to provide a richer, more complete picture of the SAP installation landscape. This makes it an invaluable tool for IT professionals and organizations seeking to optimize their SAP deployment strategies.

## Features

- **Multi-Modal Retrieval**: Combines text, images, and tables to provide comprehensive search results.
- **Agentic AI**: Utilizes AI models to understand user queries and deliver precise answers.
- **Document Embedding**: Converts documents into vector representations for efficient searching and retrieval.
- **User-Friendly Interface**: Built with Streamlit for an interactive user experience.
- **PDF Document Support**: Allows users to upload and process PDF documents for information extraction.

## Technologies Used

- **Python**: The primary programming language for the project.
- **Streamlit**: For building the web application interface.
- **LangChain**: For managing document embeddings and retrieval.
- **OpenAI API**: For leveraging advanced language models.
- **AstraDB**: For storing and retrieving vectorized documents.
- **Unstructured Partitioning**: For extracting content from PDF files.


## Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a `.env` file in the root directory of the project and add your API keys:

   ```plaintext
   OPENAI_API_KEY=<your-openai-api-key>
   GROQ_API_KEY=<your-groq-api-key>
   LANGCHAIN_API_KEY=<your-langchain-api-key>
   ASTRA_DB_API_ENDPOINT=<your-astra-db-endpoint>
   ASTRA_DB_APPLICATION_TOKEN=<your-astra-db-application-token>
   ```

3. Run the application:

   ```bash
   streamlit run main.py
   ```

## Usage

- Upload PDF documents through the provided interface.
- Ask questions related to the content of the documents.
- Receive summarized information and detailed descriptions of images.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- [LangChain](https://langchain.com/)
- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
