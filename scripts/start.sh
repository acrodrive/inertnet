#!/usr/bin/env bash
# Container entrypoint for the custom deps image on RunPod.
#
# RunPod injects the account's SSH public key as $PUBLIC_KEY. Both proxy SSH
# (ssh.runpod.io) and direct SSH ("Expose TCP Ports: 22", needed for VS Code
# Remote-SSH / scp / rsync) forward to sshd inside the container, so start it.
set -e

if [[ -n "${PUBLIC_KEY:-}" ]]; then
    mkdir -p ~/.ssh && chmod 700 ~/.ssh
    grep -qxF "$PUBLIC_KEY" ~/.ssh/authorized_keys 2>/dev/null \
        || echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
fi

mkdir -p /run/sshd
/usr/sbin/sshd -D &

exec sleep infinity
