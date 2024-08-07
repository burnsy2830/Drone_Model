FROM python:3.9

WORKDIR /model_path

RUN pip install --upgrade pip \  && pip install --no-cache-dir -r requirements.txt

CMD ["python3", "liveguesser.py"]