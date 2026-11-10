"""The computer's "Connect a phone" screen. RemoteGuard answers these routes to this computer only.

    GET  /api/phone/status   {token, public_url, app_url, app_ready, pair_link, link: {state, url, error}}
    POST /api/phone/link     {on: true|false}  start or stop the phone link (remembered for the next start)
    POST /api/phone/reset    a new code: every paired phone has to scan again
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from . import remote, tunnel
from .common import ctx

router = APIRouter()


class LinkIn(BaseModel):
    on: bool


async def _status(c, token: str) -> dict:
    url = c.public_url()
    ready = bool(url) and await run_in_threadpool(remote.app_online)
    if c.settings.public_url:
        link = {"state": "on", "url": url, "error": None}
    else:
        link = c.tunnel.status() if c.tunnel else {"state": "off", "url": None, "error": None}
    return {"token": token, "public_url": url, "app_url": remote.APP_URL, "app_ready": ready,
            "pair_link": remote.pair_link(remote.APP_URL if ready else None, url, token) if url else None,
            "link": link}


@router.get("/api/phone/status")
async def status(c=Depends(ctx)) -> dict:
    return await _status(c, remote.phone_token(c.data_dir))


@router.post("/api/phone/link")
async def link(body: LinkIn, c=Depends(ctx)) -> dict:
    tunnel.remember(c.data_dir, body.on)
    await run_in_threadpool(c.set_phone_link, body.on)
    return await _status(c, remote.phone_token(c.data_dir))


@router.post("/api/phone/reset")
async def reset(c=Depends(ctx)) -> dict:
    return await _status(c, remote.reset_token(c.data_dir))
