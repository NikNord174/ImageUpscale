FROM python:3.11-slim

WORKDIR /workdir

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/workdir

CMD ["python", "-m", "scripts.train"]
