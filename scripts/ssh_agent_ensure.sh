# Ensure a single persistent ssh-agent is running on a fixed socket and
# export SSH_AUTH_SOCK to it. Safe to source repeatedly (setup.sh, .bashrc,
# any new terminal) -- it reuses the existing agent instead of starting a
# new one. ssh-agent daemonizes/detaches itself, so it survives the
# terminal that started it closing, and has no built-in timeout unless
# `-t` is passed (not used here).
_SSH_AGENT_SOCK="/myhome/.ssh/agent.sock"

if SSH_AUTH_SOCK="$_SSH_AGENT_SOCK" ssh-add -l >/dev/null 2>&1; [ "$?" != "2" ]; then
    : # existing agent is reachable (0 = has keys, 1 = running with no keys yet)
else
    rm -f "$_SSH_AGENT_SOCK"
    eval "$(ssh-agent -s -a "$_SSH_AGENT_SOCK")" >/dev/null
    ssh-add /myhome/.ssh/id_rsa
    ssh-add /myhome/.ssh/id_ed25519
fi

export SSH_AUTH_SOCK="$_SSH_AGENT_SOCK"
unset _SSH_AGENT_SOCK
