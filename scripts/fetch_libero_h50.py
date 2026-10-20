"""Download the selected H50 LIBERO checkpoint without changing active config.

Pinned Hub revision, resumable files, streamed SHA256 checks, durable status.
This downloads data only; it does not execute repository code or load a model.
"""
import hashlib
import json
import os
from pathlib import Path

MODEL = 'lerobot/pi05_libero_finetuned_quantiles_v044'
REVISION = 'b6bfb493bc0f21c4ab24d816409ea3cebf5e43bd'
DESTINATION = Path('/home/wyk/workspace/openpi_cache/openpi-assets/checkpoints/lerobot_pi05_libero_h50_quantiles_v044')


def main():
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
    from huggingface_hub import HfApi, hf_hub_download
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = {'repo_id': MODEL, 'revision': REVISION, 'status': 'fetching_metadata', 'files': []}
    status = DESTINATION / 'download_manifest.json'

    def save():
        temporary = status.with_suffix('.tmp')
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(status)

    save()
    try:
        info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True)
        if info.sha != REVISION:
            raise RuntimeError('Hub revision mismatch')
        files = sorted((f for f in info.siblings if f.rfilename != '.gitattributes'),
                       key=lambda f: (f.rfilename == 'model.safetensors', f.rfilename))
        for item in files:
            manifest.update(status='downloading', current_file=item.rfilename)
            save()
            print('downloading=' + item.rfilename, flush=True)
            path = Path(hf_hub_download(MODEL, item.rfilename, revision=REVISION,
                                       local_dir=str(DESTINATION)))
            if path.stat().st_size != item.size:
                raise RuntimeError('File size mismatch: ' + item.rfilename)
            sha = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    sha.update(block)
            digest = sha.hexdigest()
            expected = item.lfs.sha256 if item.lfs else None
            if expected is not None and digest != expected:
                raise RuntimeError('LFS SHA256 mismatch: ' + item.rfilename)
            manifest['files'].append(dict(name=item.rfilename, size=item.size,
                                          sha256=digest, expected_lfs_sha256=expected))
            print('verified=' + item.rfilename, flush=True)
        config = json.loads((DESTINATION / 'config.json').read_text())
        train = json.loads((DESTINATION / 'train_config.json').read_text())
        if config['chunk_size'] != 50 or train['policy']['chunk_size'] != 50:
            raise RuntimeError('Expected published inference and training H50')
        manifest.update(status='download_verified', current_file=None,
                        inference_horizon=50, training_horizon=50,
                        integration_status='not_loaded_or_validated; active project config unchanged')
        save()
        print('download_verified=' + str(DESTINATION), flush=True)
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    main()
