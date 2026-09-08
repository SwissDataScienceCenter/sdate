import h5py

p = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
with h5py.File(p, "r") as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(name, obj.shape, obj.dtype)
    f.visititems(show)
    print("--- root attrs ---")
    for k, v in f.attrs.items():
        print(k, v)
    print("--- exchange attrs ---")
    if "exchange" in f:
        for k, v in f["exchange"].attrs.items():
            print(k, v)
        for dsname in f["exchange"]:
            ds = f["exchange"][dsname]
            print(dsname, "attrs:", dict(ds.attrs))
