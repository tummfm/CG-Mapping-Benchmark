#! /usr/bin/env -S bash -e

#equilibration time in Gromacs units (ps)
equi=0
echo equi = $equi

echo "Running force matching"
csg_fmatch --top ../atomistic/topol.tpr --trj /home/franz/hexane/hexane_ttot=100ns_dt=1fs_nstxout=200.trr --begin $equi --options fmatch.xml --cg hexane.xml --verbose1

csg_call table integrate bond.force bond.pot
csg_call table linearop bond.pot bond.pot -1 0

csg_call table integrate A-A.force A-A.pot
csg_call table linearop A-A.pot A-A.pot -1 0

cp bond.pot input_bond.pot
cp A-A.pot input_A-A.pot

csg_call --ia-type bond --ia-name bond --options fmatch.xml convert_potential gromacs --clean input_bond.pot table_b1.xvg
csg_call --ia-type non-bonded --ia-name A-A --options fmatch.xml convert_potential gromacs --clean input_A-A.pot table_A_A.xvg