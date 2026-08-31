# Iki asamali: derleme araclari calisma zamaninda gereksiz, imaj sisiyordu.
# Python surumu notebook'un calistigi surumle ayni (bkz. models/meta.json).
FROM python:3.14-slim AS derleme

WORKDIR /derle
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --prefix=/kurulum -r requirements-api.txt


FROM python:3.14-slim

# Root olarak calistirmiyorum.
RUN useradd --create-home --uid 10001 servis

WORKDIR /app
COPY --from=derleme /kurulum /usr/local
COPY api/ ./api/

# models/ imaja girmiyor, compose salt-okunur bagliyor. Boylece model
# degisince imaji yeniden kurmak gerekmiyor.

USER servis
EXPOSE 8000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Model yuklenmediyse /health 503 donuyor, yani konteyner saglikli gorunmuyor.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"

CMD ["uvicorn", "main:app", "--app-dir", "api", "--host", "0.0.0.0", "--port", "8000"]
