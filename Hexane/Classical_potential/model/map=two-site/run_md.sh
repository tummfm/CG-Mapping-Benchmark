
PREFIX=""

csg_call table integrate bond.force bond.pot
csg_call table linearop bond.pot bond.pot -1 0

csg_call table integrate A-A.force A-A.pot
csg_call table linearop A-A.pot A-A.pot -1 0

cp bond.pot input_bond.pot
cp A-A.pot input_A-A.pot

csg_call --ia-type bond --ia-name bond --options fmatch.xml convert_potential gromacs --clean input_bond.pot table_b1.xvg
csg_call --ia-type non-bonded --ia-name A-A --options fmatch.xml convert_potential gromacs --clean input_A-A.pot table_A_A.xvg

gmx grompp -n index.ndx -f grompp.mdp -p topol.top -o topol_cg.tpr -c conf_cg.gro --maxwarn 3
gmx mdrun -s topol.tpr -c confout.gro -o traj.trr -x traj.xtc -tableb table_b1.xvg -v 
