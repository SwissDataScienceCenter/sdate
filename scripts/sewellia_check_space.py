import shutil
u = shutil.disk_usage("/mydata/sdate/shared")
print(f"total={u.total/1e9:.1f}GB used={u.used/1e9:.1f}GB free={u.free/1e9:.1f}GB")
