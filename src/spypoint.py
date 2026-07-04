import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx

BASE_URL = "https://restapi.spypoint.com/api/v3"


@dataclass
class Camera:
    id: str
    name: str


@dataclass
class Photo:
    id: str
    taken_at: datetime
    url: str


class SpypointError(Exception):
    pass


class SpypointClient:
    def __init__(self, username: str, password: str):
        self._username = username
        self._password = password
        self._token: Optional[str] = None

    def _auth_headers(self) -> dict:
        if not self._token:
            self.login()
        return {"authorization": f"Bearer {self._token}"}

    def login(self) -> None:
        resp = httpx.post(
            f"{BASE_URL}/user/login",
            json={"username": self._username, "password": self._password},
            timeout=15,
        )
        if resp.status_code != 200:
            raise SpypointError(f"Login failed ({resp.status_code}): {resp.text}")
        self._token = resp.json()["token"]

    def get_cameras(self) -> list[Camera]:
        resp = self._get("/camera/all")
        cameras = resp.json()
        return [
            Camera(id=c["id"], name=c.get("alias") or c.get("name") or c["id"])
            for c in cameras
        ]

    def get_latest_photo(self, camera_id: str) -> Optional[Photo]:
        photos = self.get_photos(camera_id, limit=1)
        return photos[0] if photos else None

    def get_all_photos(self, camera_id: str) -> list[Photo]:
        """Fetch all photos for a camera, paging in batches of 500."""
        all_photos: list[Photo] = []
        # SpyPoint doesn't have a cursor-based pagination API — we fetch in
        # date-descending chunks using dateEnd to walk backwards.
        date_end = "2100-01-01T00:00:00.000Z"
        batch_size = 500
        seen_ids: set[str] = set()

        while True:
            batch = self.get_photos(camera_id, limit=batch_size, date_end=date_end)
            new = [p for p in batch if p.id not in seen_ids]
            if not new:
                break
            all_photos.extend(new)
            seen_ids.update(p.id for p in new)
            if len(batch) < batch_size:
                break  # last page
            # Move date_end back to just before the oldest photo in this batch
            oldest = min(new, key=lambda p: p.taken_at)
            date_end = oldest.taken_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        return sorted(all_photos, key=lambda p: p.taken_at, reverse=True)

    def get_photos(self, camera_id: str, limit: int = 10, date_end: str = "2100-01-01T00:00:00.000Z") -> list[Photo]:
        resp = self._post(
            "/photo/all",
            json={
                "camera": [camera_id],
                "dateEnd": date_end,
                "hd": False,
                "favorite": False,
                "limit": limit,
                "tag": [],
            },
        )
        raw = resp.json().get("photos") or resp.json()
        return [_parse_photo(p) for p in raw]

    def _get(self, path: str) -> httpx.Response:
        return self._request("GET", path)

    def _post(self, path: str, json: dict) -> httpx.Response:
        return self._request("POST", path, json=json)

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        resp = httpx.request(
            method,
            f"{BASE_URL}{path}",
            headers=self._auth_headers(),
            timeout=15,
            **kwargs,
        )
        if resp.status_code == 401:
            self.login()
            resp = httpx.request(
                method,
                f"{BASE_URL}{path}",
                headers=self._auth_headers(),
                timeout=15,
                **kwargs,
            )
        if not resp.is_success:
            raise SpypointError(f"{method} {path} failed ({resp.status_code}): {resp.text}")
        return resp


def _parse_photo(data: dict) -> Photo:
    large = data.get("large") or {}
    url = f"https://{large['host']}/{large['path']}" if large.get("host") else ""

    photo_id = data["id"]

    # Primary: decode from MongoDB ObjectID (first 4 bytes = Unix timestamp UTC).
    # This is more reliable than the API-provided date field.
    taken_at = _timestamp_from_object_id(photo_id)

    # Fallback: use the API-provided date if ObjectID decoding failed
    if taken_at is None:
        raw_date = data.get("date") or data.get("createdAt") or data.get("takenAt") or ""
        try:
            taken_at = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            taken_at = datetime.now(timezone.utc)

    return Photo(id=photo_id, taken_at=taken_at, url=url)


def _timestamp_from_object_id(photo_id: str) -> datetime | None:
    """Decode a UTC datetime from the first 4 bytes of a MongoDB ObjectID."""
    try:
        unix_ts = struct.unpack(">I", bytes.fromhex(photo_id[:8]))[0]
        return datetime.fromtimestamp(unix_ts, tz=timezone.utc)
    except Exception:
        return None
