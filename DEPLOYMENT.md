# Deployment

## Recommended Public Demo

Deploy `portfolio-web/` as a static site. This is the safest interview-facing path because it does not require GPU models, Redis, Celery workers, or persistent backend storage.

Options:

- **GitHub Pages:** use `.github/workflows/pages.yml` after enabling Pages with GitHub Actions as the source.
- **Vercel / Netlify:** set the project root or publish directory to `portfolio-web/`.
- **Any static host:** upload the contents of `portfolio-web/`.

Local preview:

```bash
cd portfolio-web
python -m http.server 4173
```

## Optional Full-Stack Demo

Use this only when you want a live FastAPI backend behind the portfolio page.

```bash
docker build -f docker/demo.backend.Dockerfile -t anime-adventure-lab-demo-api .
docker build -f docker/demo.Dockerfile -t anime-adventure-lab-demo .
```

The full React/FastAPI stack can also be run with Docker Compose, but GPU inference and workers should be treated as advanced/local mode unless the target host has the required model warehouse.

## Existing Portfolio Gateway

This project can still be deployed behind the portfolio gateway:

- URL: `https://neojustin.dothost.net/p/anime-adventure-lab/`
- Services:
  - UI: `anime-adventure-lab`
  - Backend demo API: `anime-adventure-lab-backend`

Update flow:

```bash
cd /home/neojustin/justin-portfolio
docker-compose up -d --build anime-adventure-lab-backend anime-adventure-lab
```

Reference:

- `/home/justin/web-projects/justin-portfolio/docs/deployment/update-workflow.md`
