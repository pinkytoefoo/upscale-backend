# Upscalerr
backend for my image upscaling website

## How to run
### Docker
```
docker compose build
docker compose up
```
### Uvicorn
```
uvicorn main:app --host 0.0.0.0 --port 8000 --workers <amount of workers>
```

## Things to work on
- [ ] Add hardware monitoring (psutil)
- [ ] Improve logging
- [ ] Adding anime / manga upscalers
- [ ] Adding api safety measures (e.g. rate limiting, worker querying, ip tracking, etc.)
- [ ] Abstracting files into easier pieces
- [ ] Working with cloudflare workers
- [ ] Setting up frontend
- [ ] Docker composing frontend with this backend
- [X] Docker

## Helpful links
Scalable fastapi
[YouTube vid](https://www.youtube.com/watch?v=Af6Zr0tNNdE&t=180s)
[Github repo for vid](https://github.com/ArjanCodes/examples/tree/main/2025/project/app)

Workers and hardware management
[Medium article on gunicorn](https://medium.com/@iklobato/mastering-gunicorn-and-uvicorn-the-right-way-to-deploy-fastapi-applications-aaa06849841e)
