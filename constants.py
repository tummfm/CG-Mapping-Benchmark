Dataset_paths = {
    'hexane': 'Reference_simulations/hexane/hexane_ttot=100ns_dt=1fs_nstxout=200.npz',
    'ala2': '/home/franz/l-ala2_ttot=500ns_dt=0.5fs_nstxout=2000.npz',
    'pro2': '/home/franz/l-pro2_ttot=500ns_dt=0.5fs_nstxout=2000.npz',
    'thr2': '/home/franz/l-thr2_ttot=500ns_dt=0.5fs_nstxout=2000.npz',
    'gly2': '/home/franz/l-gly2_ttot=500ns_dt=0.5fs_nstxout=2000.npz',
    'ala15': '/home/franz/Lala15_ttot=500ns_dt=0.5fs.npz',
}
def _get_available_datasets():
    return list(Dataset_paths.keys())
