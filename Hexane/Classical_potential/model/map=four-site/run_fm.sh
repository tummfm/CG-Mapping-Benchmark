# ! /usr/bin/env -S bash -e

#equilibration time in Gromacs units (ps)
equi=0
echo equi = $equi

echo "Running force matching"
csg_fmatch --top ../atomistic/topol.tpr --trj /home/franz/hexane/hexane_ttot=100ns_dt=1fs_nstxout=200.trr --begin $equi --options fmatch.xml --cg hexane.xml --verbose1
