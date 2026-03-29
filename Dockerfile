FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 7860

ENV PYTHONPATH=/app
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "api.app:app"]