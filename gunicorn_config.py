bind = "0.0.0.0:10000"  # Render utilise le port interne 10000
workers = 2  # Suffisant pour l'instance gratuite
worker_class = "uvicorn.workers.UvicornWorker"