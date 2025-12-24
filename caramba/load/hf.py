"""
hf provides Hugging Face download utilities.
"""
from __future__ import annotations

import json
from pathlib import Path


_DEFAULT_FILES = [
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "pytorch_model.bin",
]


class HfDownload:
    """
    HfDownload downloads a file from Hugging Face Hub.
    """
    def __init__(
        self,
        *,
        repo_id: str,
        filename: str | None,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        """
        __init__ initializes the Hugging Face download spec.
        """
        if not repo_id:
            raise ValueError("repo_id must be non-empty")
        if filename is not None and not filename:
            raise ValueError("filename must be non-empty")
        self.repo_id: str = repo_id
        self.filename: str | None = filename
        self.revision: str | None = revision
        self.cache_dir: str | None = cache_dir

    @classmethod
    def from_uri(cls, uri: str) -> HfDownload:
        """
        from_uri parses an hf:// URI into an HfDownload.
        """
        if not uri.startswith("hf://"):
            raise ValueError(f"Expected hf:// URI, got {uri!r}")
        raw = uri[len("hf://") :]
        revision = None
        if "@" in raw:
            raw, revision = raw.rsplit("@", 1)
            if not revision:
                raise ValueError(f"hf:// URI has empty revision: {uri!r}")
        parts = [p for p in raw.split("/") if p]
        if len(parts) < 2:
            raise ValueError(
                "hf:// URI must include repo_id, got "
                f"{uri!r}"
            )
        if len(parts) == 2:
            repo_id = "/".join(parts)
            filename = None
        else:
            repo_id = "/".join(parts[:2])
            filename = "/".join(parts[2:])
        return cls(repo_id=repo_id, filename=filename, revision=revision)

    def fetch(self) -> Path:
        """
        fetch downloads the file and returns the local path.
        """
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for hf:// downloads. "
                "Install it with `pip install huggingface_hub`."
            ) from exc

        filename = self.filename
        if filename is None:
            api = HfApi()
            repo_files = api.list_repo_files(
                repo_id=self.repo_id,
                revision=self.revision,
            )
            filename = self._select_default_file(repo_files)

        path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            revision=self.revision,
            cache_dir=self.cache_dir,
        )
        out = Path(path)
        if not out.is_file():
            raise ValueError(f"Downloaded file not found: {out}")

        if out.name.endswith(".index.json"):
            self._download_index_shards(out)
        return out

    def _select_default_file(self, repo_files: list[str]) -> str:
        """
        _select_default_file picks a default weight file from repo files.
        """
        for candidate in _DEFAULT_FILES:
            if candidate in repo_files:
                return candidate
        raise ValueError(
            "No supported checkpoint file found. "
            f"Expected one of: {_DEFAULT_FILES}"
        )

    def _download_index_shards(self, index_path: Path) -> None:
        """
        _download_index_shards ensures all index shards are downloaded.
        """
        data = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError("Invalid index file: missing weight_map")
        shards = sorted({str(v) for v in weight_map.values()})

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for hf:// downloads. "
                "Install it with `pip install huggingface_hub`."
            ) from exc

        for shard in shards:
            hf_hub_download(
                repo_id=self.repo_id,
                filename=shard,
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
