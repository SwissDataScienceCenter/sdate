rm -r /root/.claude
ln -s /myhome/.claude /root/.claude

rm -r /root/.ssh
ln -s /myhome/.ssh /root/.ssh

source /myhome/sdate/scripts/ssh_agent_ensure.sh

# sshd logs in as root, so its StrictModes check requires ~/.ssh (-> /myhome/.ssh)
# to be root-owned; a wipe/recreate of the persistent .ssh folder tends to leave
# it owned by jovyan instead, which makes sshd silently refuse every key and
# fall back to password auth.
chown root:root /myhome/.ssh

# /myhome itself is a shared, world-writable mount owned by jovyan (Jupyter/etc),
# which independently fails the same StrictModes check. Disabling StrictModes
# (rather than chown/chmod-ing the shared mount) avoids touching /myhome itself.
sed -i 's/^#\?StrictModes.*/StrictModes no/' /etc/ssh/sshd_config
grep -q '^StrictModes no' /etc/ssh/sshd_config || echo 'StrictModes no' >> /etc/ssh/sshd_config

# The real sshd is started by the image's own /entrypoint.sh at container boot,
# *before* this script ever runs -- so by the time the edits above land, that
# already-running sshd has the old (broken) config cached in memory. Editing
# the file alone does nothing for it; it must be told to re-read the file.
#
# IMPORTANT: match only sshd's own self-renamed process title ("sshd: ..."),
# never a plain `-f '/usr/sbin/sshd -D'` substring match -- both tini (PID 1,
# the container's init) and the bash running /entrypoint.sh also carry that
# exact string in THEIR OWN command line (as the argument they were launched
# with), so a broad match kill -HUP's tini too and takes the whole container
# down with it. Confirmed the hard way: it did exactly that.
pgrep -f '^sshd: /usr/sbin/sshd -D \[listener\]' | xargs -r kill -HUP 

export HOME=/myhome
export PATH="/opt/conda/bin:$PATH"
export CLAUDE_CONFIG_DIR=/myhome/.claude
export XDG_CACHE_HOME=/myhome/.cache
wandb login d6f99b98acf9c1a284aa2ba5830f3eca60fde2f0

# The Claude Code standalone install (used by the VSCode extension) places its
# binary at /root/.local/bin/claude (symlinked to a versioned install under
# /root/.local/share/claude), but that directory isn't on PATH by default, so
# `claude` isn't callable in a plain terminal. /root/.bashrc resets with the
# ephemeral rootfs on every restart, so re-add it here each time instead of
# relying on a one-off edit to survive.
grep -q '\.local/bin' /root/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> /root/.bashrc

python -m pip install joblib
python -m pip install ipywidgets
python -m pip install pandas

# sdate.stream_hvec (HEVC gray10 movie encoding, used by write_projection_movie
# and friends) shells out to the `ffmpeg` binary -- not bundled in this image.
# apt-installed ffmpeg only lands in the container's ephemeral rootfs, so every
# fresh job would otherwise re-run apt-get from scratch. Instead, cache the
# binary (+ its shared-lib closure) on persistent storage (/myhome) the first
# time any job needs it, then every later job just gets it prepended to PATH --
# no reinstall. The lib closure is only exposed to ffmpeg itself via a wrapper
# script's LD_LIBRARY_PATH, not exported for the whole job, so it can't shadow
# libraries other code (e.g. torch/CUDA) depends on.
FFMPEG_HOME=/myhome/tools/ffmpeg
if [ -x "$FFMPEG_HOME/bin/ffmpeg" ]; then
    export PATH="$FFMPEG_HOME/bin:$PATH"
elif ! command -v ffmpeg >/dev/null 2>&1; then
    (apt-get update -qq && apt-get install -y -qq ffmpeg) || (conda install -y -c conda-forge ffmpeg -q)
    if command -v ffmpeg >/dev/null 2>&1; then
        FFBIN="$(command -v ffmpeg)"
        mkdir -p "$FFMPEG_HOME/bin" "$FFMPEG_HOME/lib"
        cp "$FFBIN" "$FFMPEG_HOME/ffmpeg.bin"
        ldd "$FFBIN" 2>/dev/null | awk '/=> \// {print $3}' | sort -u | xargs -I{} cp -n {} "$FFMPEG_HOME/lib/" 2>/dev/null
        cat > "$FFMPEG_HOME/bin/ffmpeg" <<'EOF'
#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LD_LIBRARY_PATH="$DIR/lib:$LD_LIBRARY_PATH"
exec "$DIR/ffmpeg.bin" "$@"
EOF
        chmod +x "$FFMPEG_HOME/bin/ffmpeg"
        export PATH="$FFMPEG_HOME/bin:$PATH"
    fi
fi

# LaTeX (pdflatex + the packages the smartt paper.tex needs: siunitx,
# tikz/pgf, booktabs, multirow, authblk, natbib, hyperref, ...) is only in
# the container's ephemeral rootfs, so it disappears on every restart, same
# as ffmpeg above. Unlike ffmpeg it isn't a single relocatable binary (TeX
# Live hardcodes absolute paths across /usr/share/texlive, /usr/share/texmf*
# and /var/lib/texmf), so it's just re-installed via apt each time rather
# than cached on persistent storage -- only pdflatex itself is checked
# so a restart doesn't re-run apt needlessly.
# DEBIAN_FRONTEND=noninteractive is required: without it, tzdata's postinst
# script prompts for a timezone on a fresh container and hangs forever with
# no TTY to answer it (confirmed 2026-09-04 -- a plain `apt-get install -y`
# sat stuck on tzdata's debconf prompt for 15+ minutes until killed).
if ! command -v pdflatex >/dev/null 2>&1; then
    DEBIAN_FRONTEND=noninteractive apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        texlive-latex-base texlive-latex-recommended texlive-latex-extra \
        texlive-fonts-recommended texlive-pictures texlive-science \
        texlive-bibtex-extra latexmk
fi

# python -m pip install monai

# cd /myhome/BaseTraining/
# python -m pip install -e .
# cd /myhome/astra-torch/
# python -m pip install -e .
# cd /myhome/DiffusionBase
# python -m pip install -e .


# cd /myhome/chip-project
# python -m pip install -e .

cd /myhome/sdate
# --no-deps: register the sdate package as importable/editable without letting
# pip re-resolve its (unpinned in setup.py) third-party deps every job -- that
# resolution was pulling in a torch/diffusers combination incompatible enough
# to break diffusers' own import (missing torch.xpu, then torch.float8_e4m3fn)
# on every fresh container, regardless of which script runs.
python -m pip install -e . --no-deps


# Reuse a single long-lived ssh-agent bound to a fixed socket instead of
# spawning a fresh one per shell/terminal (agents never time out and were
# accumulating as orphans, one per terminal ever opened).

git config --global user.email "lfbarba@gmail.com"
git config --global user.name "Luis Barba"


# start ssh server
/usr/sbin/sshd