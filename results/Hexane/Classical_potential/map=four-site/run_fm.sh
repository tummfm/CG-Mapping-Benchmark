# ! /usr/bin/env -S bash -e

#equilibration time in Gromacs units (ps)
equi=0
echo equi = $equi

echo "Running force matching"
csg_fmatch --top ../atomistic/topol.tpr --trj /home/franz/hexane/hexane_ttot=100ns_dt=1fs_nstxout=200.trr --begin $equi --options fmatch.xml --cg hexane.xml --verbose1

csg_call table integrate bond.force bond.pot
csg_call table linearop bond.pot bond.pot -1 0

csg_call table integrate angle.force angle.pot
csg_call table linearop angle.pot angle.pot -1 0

csg_call table integrate dihedral.force dihedral.pot
csg_call table linearop dihedral.pot dihedral.pot -1 0

csg_call table integrate A-A.force A-A.pot
csg_call table linearop A-A.pot A-A.pot -1 0

csg_call table integrate B-B.force B-B.pot
csg_call table linearop B-B.pot B-B.pot -1 0

csg_call table integrate A-B.force A-B.pot
csg_call table linearop A-B.pot A-B.pot -1 0

cp bond.pot input_bond.pot
cp angle.pot input_angle.pot
cp dihedral.pot input_dihedral.pot
cp A-A.pot input_A-A.pot
cp A-B.pot input_A-B.pot
cp B-B.pot input_B-B.pot

csg_call --ia-type bond --ia-name bond --options fmatch.xml convert_potential gromacs --clean input_bond.pot table_b1.xvg
csg_call --ia-type angle --ia-name angle --options fmatch.xml convert_potential gromacs --clean input_angle.pot table_a1.xvg
csg_call --ia-type dihedral --ia-name dihedral --options fmatch.xml convert_potential gromacs --clean input_dihedral.pot table_d1.xvg
csg_call --ia-type non-bonded --ia-name A-A --options fmatch.xml convert_potential gromacs --clean input_A-A.pot table_A_A.xvg
csg_call --ia-type non-bonded --ia-name A-B --options fmatch.xml convert_potential gromacs --clean input_A-B.pot table_A_B.xvg
csg_call --ia-type non-bonded --ia-name B-B --options fmatch.xml convert_potential gromacs --clean input_B-B.pot table_B_B.xvg