# GenAI-hackathon-project

## Healthcare Data Query System
This is a Streamlit-based application that allows users to query healthcare data using natural language. The system uses OpenAI's GPT-4 to convert natural language queries into SQL and provides appropriate visualizations of the results.

# Features

- Natural language query processing
- Automatic SQL query generation
- Multiple output formats (text, tables, plots)
- Secure API key handling
- Interactive web interface
- Efficient database management
- Real-time query processing

# Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Streamlit account (for deployment)

# Installation

Clone the repository:

- git clone https://github.com/yourusername/healthcare-query-system.git
- cd healthcare-query-system

Install required packages:

- pip install -r requirements.txt

Place your healthcare dataset (CSV file) in the project root directory and ensure it's named Healthcare_dataset_with_summary.csv

Usage

Start the Streamlit application:

- streamlit run main.py

- Open your web browser and navigate to http://localhost:8501 (or the URL provided by Streamlit)
- Enter your query in natural language, for example:

"How many patients are diabetic?"
"What is the average age of patients?"
"Show me the distribution of blood pressure readings"


Click "Submit Query" to see the results




