"""Recreate the CUDA/FlashInfer compatibility fixes used by the H20 quality host."""
from pathlib import Path
import importlib.util
import re
import sysconfig

site = Path(sysconfig.get_paths()['purelib'])
cuda = site/'nvidia/cu13'
lib = cuda/'lib'
if not lib.is_dir():
    raise RuntimeError(f'CUDA 13 libraries missing: {lib}')
if not (cuda/'lib64').exists():
    (cuda/'lib64').symlink_to('lib', target_is_directory=True)
for target in sorted(lib.glob('lib*.so.*')):
    name = target.name.split('.so.')[0]+'.so'
    link = lib/name
    if not link.exists():
        link.symlink_to(target.name)
spec = importlib.util.find_spec('flashinfer')
if spec is None or spec.origin is None:
    raise RuntimeError('flashinfer is not installed')
path = Path(spec.origin).parent/'comm/cuda_ipc.py'
old = 'if lib_name in line:'
new = 'if lib_name in line and not (lib_name == "libcudart" and "libcudart_stub" in line):'
text = path.read_text()
if new not in text:
    if text.count(old) != 1:
        raise RuntimeError('Unexpected FlashInfer CUDA IPC implementation; inspect before patching')
    path.write_text(text.replace(old, new, 1))
print('CUDA development links and FlashInfer libcudart lookup ready')
