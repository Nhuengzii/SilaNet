FROM pytorch/pytorch

RUN pip install gradio
RUN mkdir app
WORKDIR /app
RUN mkdir checkpoint
COPY ./checkpoint/generator_sila_1000.pt /app/checkpoint/
COPY ./models.py /app/models.py
COPY ./app.py /app/app.py

CMD ["python", "app.py"]
