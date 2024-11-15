# Use LangChain's base image
FROM langchain/langchain

# Set the working directory
WORKDIR /app

# Install any necessary dependencies for Streamlit and system libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install required Python packages
COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

# Copy the application code into the container
COPY csv_bot.py .
COPY utils.py .
COPY chains.py .

# Expose the Streamlit default port
EXPOSE 8503

# Add a health check to monitor container health
HEALTHCHECK CMD curl --fail http://localhost:8503/_stcore/health || exit 1

# Start Streamlit
ENTRYPOINT ["streamlit", "run", "csv_bot.py", "--server.port=8503", "--server.address=0.0.0.0"]
