#!/bin/bash
# This script is used to run a GROMACS simulation for the ACE-ala_ala_ala-NME system.
# It includes steps for preparing the system, energy minimization, equilibration, and production MD.

GPU_ID=0
NTMPI=1
NTOMP=16

# Generate box and solvation
/usr/local/gromacs/bin/gmx pdb2gmx -f coords/ace_L-ala15_nme.pdb -o coords/ace_L-ala15_nme.gro -ff amber99sb-ildn -water tip3p -p top/topol.top
mv posre.itp top/
/usr/local/gromacs/bin/gmx editconf -f coords/ace_L-ala15_nme.gro -o box/ace_L-ala15_nme_box.gro -c -d 1.2 -bt cubic
/usr/local/gromacs/bin/gmx solvate -cp box/ace_L-ala15_nme_box.gro -cs spc216.gro -o solvation/ace_L-ala15_nme_solv.gro -p top/topol.top

# Adding ions (Not neccessary in this case, but included for completeness)
/usr/local/gromacs/bin/gmx grompp -f solvation/ions.mdp -c solvation/ace_L-ala15_nme_solv.gro -p top/topol.top -o solvation/ions.tpr
/usr/local/gromacs/bin/gmx genion -s solvation/ions.tpr -o solvation/solv_ions.gro -p top/topol.top -pname K -nname CL -neutral

# Energy minimization
/usr/local/gromacs/bin/gmx grompp -f emin/minim.mdp -c solvation/solv_ions.gro -p top/topol.top -o emin/em.tpr -maxwarn 1
/usr/local/gromacs/bin/gmx mdrun -v -deffnm emin/em -gpu_id $GPU_ID -ntmpi $NTMPI -ntomp $NTOMP

# NVT equilibration
/usr/local/gromacs/bin/gmx grompp -f equil/nvt.mdp -c emin/em.gro -r emin/em.gro -p top/topol.top -o equil/nvt.tpr
/usr/local/gromacs/bin/gmx mdrun -v -deffnm equil/nvt -gpu_id $GPU_ID -ntmpi $NTMPI -ntomp $NTOMP

# NPT equilibration
/usr/local/gromacs/bin/gmx grompp -f equil/npt.mdp -c equil/nvt.gro -r equil/nvt.gro -t equil/nvt.cpt -p top/topol.top -o equil/npt.tpr
/usr/local/gromacs/bin/gmx mdrun -v -deffnm equil/npt -gpu_id $GPU_ID -ntmpi $NTMPI -ntomp $NTOMP

# Production MD
/usr/local/gromacs/bin/gmx grompp -f MD/md.mdp -c equil/npt.gro -t equil/npt.cpt -p top/topol.top -o MD/md_0_1.tpr
/usr/local/gromacs/bin/gmx mdrun -deffnm MD/md_0_1 -v -gpu_id $GPU_ID -ntmpi $NTMPI -ntomp $NTOMP

nohup /usr/local/gromacs/bin/gmx mdrun -deffnm MD/md_0_1 -gpu_id 0 -ntmpi 1 -ntomp 16 &