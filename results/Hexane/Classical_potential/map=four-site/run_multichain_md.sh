#!/bin/bash

NCHAINS=10

# Loop over all chains
for (( i=0; i<${NCHAINS}; i++ )); do
    CHAIN_DIR="chain_${i}"
    echo "=== Running chain ${i} in ${CHAIN_DIR} ==="

    if [ ! -d "${CHAIN_DIR}" ]; then
        echo "Directory ${CHAIN_DIR} not found! Skipping."
        continue
    fi

    # ----- GROMPP -----
    gmx grompp \
        -n index.ndx \
        -f grompp.mdp \
        -p topol.top \
        -c "${CHAIN_DIR}/cg_hexane.gro" \
        -o "${CHAIN_DIR}/topol_cg.tpr" \
        -po "${CHAIN_DIR}/grompp_out.mdp" \
        -pp "${CHAIN_DIR}/processed.top" \
        -maxwarn 3

    if [ $? -ne 0 ]; then
        echo "grompp failed for ${CHAIN_DIR}! Skipping mdrun."
        continue
    fi

    # ----- MDRUN -----
    gmx mdrun \
        -s "${CHAIN_DIR}/topol_cg.tpr" \
        -c "${CHAIN_DIR}/cg_hexane_out.gro" \
        -o "${CHAIN_DIR}/traj.trr" \
        -x "${CHAIN_DIR}/traj.xtc" \
        -e "${CHAIN_DIR}/ener.edr" \
        -g "${CHAIN_DIR}/md.log" \
        -cpo "${CHAIN_DIR}/state.cpt" \
        -tableb table_b1.xvg table_a1.xvg table_d1.xvg \
        -v

    echo "=== Finished chain ${i} ==="
    echo
done

echo "All chains processed."
