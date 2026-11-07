"""WP9 server (DESIGN §2, §11): FastAPI app, one spawned GPU worker, job queue, SSE, library.

Keep this file free of imports: the spawned GPU worker imports `nathan_chordify.server.worker`,
and that must not pull in FastAPI. Use `nathan_chordify.server.app.create_app()`.
"""
