FROM python:3.9
WORKDIR /model_path
RUN  MKDIR -p /var/uploads
RUN pip install --upgrade pip \  && pip install --no-cache-dir -r requirements.txt
CMD ["python3", "liveguesser.py"]