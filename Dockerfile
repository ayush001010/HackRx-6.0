# 1. Base image
FROM python:3.11-slim

# 2. Create a working directory
WORKDIR /app

# 3. Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your code
COPY . .

# 5. Expose port 8000
EXPOSE 8000

# 6. Launch command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


