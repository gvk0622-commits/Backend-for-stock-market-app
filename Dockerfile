# 1. Use the official, lightweight Python image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first (this caches the installation step)
COPY requirements.txt .

# 4. Install all the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your backend code into the container
COPY . .

# 6. Expose port 5000 for the internet to connect to
EXPOSE 5000

# 7. Run the app using Gunicorn (Production Server) instead of Flask's dev server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]