export XDG_CACHE_HOME=/myhome/.cache
wandb login d6f99b98acf9c1a284aa2ba5830f3eca60fde2f0
cd /myhome/sdate/
python -m pip install -e .
python -m pip install joblib


eval "$(ssh-agent -s)"
ssh-add /myhome/.ssh/id_rsa
ssh-add /myhome/.ssh/id_ed25519
git config --global user.email "lfbarba@gmail.com"
git config --global user.name "Luis Barba"


# start ssh server
/usr/sbin/sshd