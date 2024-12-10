import streamlit as st
import pandas as pd
import sqlite3
from openai import OpenAI
import json
import re
from contextlib import contextmanager

class DataQuerySystem:
    def __init__(self, csv_file_path, database_path='data.db'):
        """
        Initialize the DataQuerySystem with CSV data and OpenAI API key from secrets.
        """
        self.database_path = database_path
        
        # Get API key from Streamlit secrets
        self.api_key = st.secrets["openai"]["api_key"]
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize database
        self.setup_database(csv_file_path)

    @contextmanager
    def get_db_connection(self):
        """
        Context manager for database connections.
        """
        conn = sqlite3.connect(self.database_path, check_same_thread=False)
        try:
            yield conn
        finally:
            conn.close()

    def setup_database(self, csv_file_path):
        """
        Set up SQLite database from CSV file.
        """
        try:
            df = pd.read_csv(csv_file_path, low_memory=False)
            with self.get_db_connection() as conn:
                # Set timeout and isolation level
                conn.execute("PRAGMA busy_timeout = 10000")  # 10 second timeout
                conn.execute("PRAGMA journal_mode = WAL")    # Write-Ahead Logging
                conn.execute("PRAGMA synchronous = NORMAL")  # Faster synchronization
                
                # Create table and insert data
                df.to_sql('patients', conn, if_exists='replace', index=False)
                conn.commit()
        except Exception as e:
            st.error(f"Error setting up database: {str(e)}")
            raise

    def analyze_prompt(self, user_query):
        """
        Analyze user query to determine output type.
        """
        prompt = f"""You are a data analysis expert. Analyze the following user query and determine what type of output would be most appropriate.

        User Query: {user_query}

        Guidelines:
        - If the user is asking about a single record or piece of information, respond with 'string'
        - If the user is asking about multiple records or needs tabular data, respond with 'dataframe'
        - If the user is requesting any kind of visualization or comparison, respond with 'plot'

        Please respond in the following JSON format:
        {{
            "output_type": "string|dataframe|plot",
            "explanation": "Brief explanation of why this output type was chosen"
        }}
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            return {
                "output_type": "string",
                "explanation": "Failed to parse response, defaulting to string output"
            }

    def generate_sql_query(self, question):
        """
        Generate SQL query using AI model.
        """
        prompt = f"""You are an expert SQL developer. Generate a SQL query to answer the following question.
        The database contains a table named 'patients' with healthcare data.
        Question: {question}
        Return only the SQL query without any explanations or markdown.
        """

        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        query = response.choices[0].message.content.strip()
        query = re.sub(r'^```sql\n|\n```$', '', query)
        return query

    def execute_query(self, query):
        """
        Execute SQL query and return results.
        """
        with self.get_db_connection() as conn:
            try:
                return pd.read_sql_query(query, conn)
            except sqlite3.Error as e:
                st.error(f"Database error: {str(e)}")
                raise
            except pd.io.sql.DatabaseError as e:
                st.error(f"Query execution error: {str(e)}")
                raise

    def format_result(self, result, output_type):
        """
        Format the result based on output type.
        """
        if output_type == 'string':
            if len(result) == 1 and len(result.columns) == 1:
                return str(result.iloc[0, 0])
            return result.to_string()
        elif output_type == 'dataframe':
            return result
        else:
            return result

    def process_query(self, user_query):
        """
        Main function to process user queries.
        """
        try:
            # First, analyze the query type
            analysis = self.analyze_prompt(user_query)
            output_type = analysis['output_type']

            # If it's a plot request, return early
            if output_type == 'plot':
                return {
                    'type': 'plot',
                    'message': 'Plot request detected. Please handle with visualization logic.',
                    'query': user_query
                }

            # Generate and execute SQL query
            sql_query = self.generate_sql_query(user_query)
            result = self.execute_query(sql_query)

            # Format result based on output type
            formatted_result = self.format_result(result, output_type)

            return {
                'type': output_type,
                'result': formatted_result,
                'sql_query': sql_query
            }

        except Exception as e:
            return {
                'type': 'error',
                'message': f"Error processing query: {str(e)}"
            }


def initialize_session_state():
    """
    Initialize session state variables
    """
    if 'query_system' not in st.session_state:
        try:
            csv_file_path = "Healthcare_dataset_with_summary.csv"
            st.session_state.query_system = DataQuerySystem(csv_file_path)
        except Exception as e:
            st.error(f"Failed to initialize query system: {str(e)}")
            raise


def display_query_interface():
    """
    Display the query interface and handle user input
    """
    st.title("Healthcare Data Query System")
    st.write("Enter your query about the healthcare dataset.")
    
    # Input box for user query
    user_query = st.text_input("Enter your query:", key="query_input")
    
    # Submit button
    if st.button("Submit Query", key="submit_button"):
        if user_query.strip():
            with st.spinner("Processing query..."):
                result = st.session_state.query_system.process_query(user_query)
                
                # Display SQL query in expandable section
                with st.expander("View SQL Query"):
                    if 'sql_query' in result:
                        st.code(result['sql_query'], language='sql')
                
                # Display results based on type
                if result['type'] == 'error':
                    st.error(result['message'])
                elif result['type'] == 'string':
                    st.success("Query Result:")
                    st.text(result['result'])
                elif result['type'] == 'dataframe':
                    st.success("Query Result:")
                    st.dataframe(result['result'])
                elif result['type'] == 'plot':
                    st.info("Visualization request detected - Please implement visualization logic")
        else:
            st.warning("Please enter a query.")


def main():
    """
    Main application function
    """
    try:
        # Initialize session state
        initialize_session_state()
        
        # Display the query interface
        display_query_interface()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()


if __name__ == "__main__":
    main()