FROM agrigorev/zoomcamp-model:mlops-3.9.7-slim

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pip install -U pip
RUN pip install pipenv
RUN pipenv install --system --deploy

COPY ["starter.py", "./"]

# starter.py works with local filepaths, so data directory should be mounted into the container
ENTRYPOINT [ "python", "starter.py" ]