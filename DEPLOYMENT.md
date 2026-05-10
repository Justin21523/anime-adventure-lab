# Deployment

This project is deployed as a Docker service behind the portfolio gateway:

- URL: `https://neojustin.dothost.net/p/anime-adventure-lab/`
- Services (docker-compose):
  - UI: `anime-adventure-lab`
  - Backend (FastAPI demo API): `anime-adventure-lab-backend`

## Update after code changes
1) Sync code to the server checkout:
- Remote path: `/home/neojustin/justin-portfolio/projects/anime-adventure-lab`

2) Rebuild + restart on the server:
```bash
cd /home/neojustin/justin-portfolio
docker-compose up -d --build anime-adventure-lab-backend anime-adventure-lab
```

Reference workflow:
- `/home/justin/web-projects/justin-portfolio/docs/deployment/update-workflow.md`
