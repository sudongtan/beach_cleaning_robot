"""gpu_session.sh takes the Isaac flags from PID 1's environment when the ssh shell lacks them
(GPU session 1, 2026-11-13: vast.ai ssh shells don't inherit the container's --env)."""

import os
import subprocess

from conftest import ISAAC

SCRIPT = ISAAC / 'tools' / 'gpu_session.sh'
KEYS = ('ACCEPT_EULA', 'PRIVACY_CONSENT', 'OMNI_KIT_ALLOW_ROOT', 'NVIDIA_DRIVER_CAPABILITIES')


def _run(tmp_path, environ, shell_env):
    f = tmp_path / 'environ'
    f.write_bytes(b'\0'.join(f'{k}={v}'.encode() for k, v in environ.items()) + b'\0')
    env = {k: v for k, v in os.environ.items() if k not in KEYS}
    env.update(shell_env, INIT_ENVIRON=str(f), ENV_ONLY='1')
    out = subprocess.run(['bash', str(SCRIPT), 'x', '0'], env=env, capture_output=True, text=True, check=True)
    return dict(line.split('=', 1) for line in out.stdout.split())


def test_flags_taken_from_pid1(tmp_path):
    got = _run(tmp_path, {'ACCEPT_EULA': 'Y', 'PRIVACY_CONSENT': 'Y', 'OMNI_KIT_ALLOW_ROOT': '1',
                          'NVIDIA_DRIVER_CAPABILITIES': 'all', 'OTHER': 'no'}, {})
    assert got == {'ACCEPT_EULA': 'Y', 'PRIVACY_CONSENT': 'Y', 'OMNI_KIT_ALLOW_ROOT': '1',
                   'NVIDIA_DRIVER_CAPABILITIES': 'all'}


def test_shell_values_win_and_nothing_invented(tmp_path):
    got = _run(tmp_path, {'NVIDIA_DRIVER_CAPABILITIES': 'all'}, {'NVIDIA_DRIVER_CAPABILITIES': 'graphics'})
    assert got == {'NVIDIA_DRIVER_CAPABILITIES': 'graphics'}     # no EULA unless the instance had it
