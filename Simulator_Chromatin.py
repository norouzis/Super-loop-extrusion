import openmm as mm
#import openmm.app as app
import openmm.unit as unit
import numpy as np

import sys

if len(sys.argv) != 4:  # sys.argv[0] is the script name, so expecting 5 arguments
    print("Usage: python script.py crd cnd out")
    sys.exit(1)

inpcrd, incons, out = sys.argv[1:4]  # Extract filenames from command-line arguments

outrst = out + '.rst'
outxyz = out + '.xyz'
outpdb = out + '.pdb'


#from sys import stdout
#from InOut import outpdb, outrst, outxyz, inpcrd, incons
from Pars import total_time_ps, integration_step, minz_steps, minz_tol, friction, err_tol
from Pars import mass, temperature, polymer_density, bond_k, num_beads, custom_force
from Pars import bondStretchStatus, bondBendStatus, NewInteraction, CustBndForce
from Pars import n_skip_neighbors, n_cons_angles, PDB_Precision, PDB_Conversion

PI = 3.14159265359

# Simplified extract_energy function that just calculates energies
def extract_energy(state, system, context, force_group_map):
    """Extracts total potential energy and individual force contributions."""
    total_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    
    # Initialize dictionary to store energy components
    forces = {name: 0.0 for name in force_group_map}
    
    # Extract individual force contributions using assigned groups
    for force_name, group_id in force_group_map.items():
        state_component = context.getState(getEnergy=True, groups={group_id})
        forces[force_name] = state_component.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    
    return total_energy, forces

def write_pdb_frame(frame_num, state, conv, time, chain, resid, prec):
    """Writes the current frame to a PDB file."""
    pos_in_nm = state.getPositions(asNumpy=True)  # These are still OpenMM Quantity objects
    
    # Convert to unitless numpy array (assuming nanometers)
    pos_in_nm = pos_in_nm / unit.nanometer  # Removes units, now plain numbers

    with open(outpdb, "a") as f_pdb, open(outxyz, "a") as f_xyz:
        f_pdb.write(f"MODEL     {frame_num}\n")
        f_xyz.write(f"{frame_num}   {time}   {-10000}\n")

        for jj in range(len(pos_in_nm)):
            x, y, z = pos_in_nm[jj] * conv  # These are now just floats

            if prec == 3:
                f_pdb.write(f"ATOM  {jj+1:5d}  AR   AR    {resid[jj]:2d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
            elif prec == 2:
                f_pdb.write(f"ATOM  {jj+1:5d}  AR   AR    {resid[jj]:2d}    {x:8.2f}{y:8.2f}{z:8.2f}  1.00  0.00\n")
            else:
                f_pdb.write(f"ATOM  {jj+1:5d}  AR   AR    {resid[jj]:2d}    {x:8.1f}{y:8.1f}{z:8.1f}  1.00  0.00\n")

            f_xyz.write(f"{x:12.3f} {y:12.3f} {z:12.3f}\n")

        f_pdb.write("ENDMDL\n")

def writerst(state, data):
    """Writes restart file (RST) for the simulation."""
    pos_in_nm = state.getPositions(asNumpy=True)  # These are Quantity objects

    # Convert to unitless (nanometers)
    pos_in_nm = pos_in_nm / unit.nanometer

    with open(outrst, "w") as f_rst:
        for ii in range(len(pos_in_nm)):
            f_rst.write(f"{int(data[ii][0]):6d} {pos_in_nm[ii][0]:10.3f} {pos_in_nm[ii][1]:10.3f} {pos_in_nm[ii][2]:10.3f} "
                        f"{data[ii][4]:7.3f} {data[ii][5]:7.3f} {data[ii][6]:7.3f} {data[ii][7]:7.3f} "
                        f"{data[ii][8]:8.5f} {data[ii][9]:8.3f} {data[ii][10]:7.3f} {int(data[ii][11]):6d}\n")

def parse_chrom_info(chrom_type, chrom_coor, chrom_e, init_coor, resid, N):
    """Parses chromatin interaction information."""
    for ii in range(N):
        ist, ien = chrom_coor[ii]
        strength = chrom_e[ii]

        if chrom_type[ii] == 'C':
            for k in range(ist, ien+1):
                init_coor[k][7] = strength
                init_coor[k][11] = ii + 2
                resid[k] = ii + 2
        elif chrom_type[ii] == 'W':
            for k in range(ist, ien+1):
                init_coor[k][11] = ii + 30
                resid[k] = ii + 30
        elif chrom_type[ii] == 'H':
            init_coor[ist][11] = int(strength)
            resid[ist] = int(strength)
            init_coor[ien][11] = int(strength)
            resid[ien] = int(strength)

def apply_custom_force(system, bond_stretch, custom_force, resid, init_coor):
    """Applies different custom forces based on the specified type."""
    
    if custom_force == 'w':  # Spherical ball
        wall = mm.CustomExternalForce("1*max(0, r-450.0)^2; r=sqrt(x*x+y*y+z*z)")
        system.addForce(wall)
        for ii in range(system.getNumParticles()):
            wall.addParticle(ii)

    elif custom_force == 'h':  # Hemi-spherical ball
        wall = mm.CustomExternalForce("1*max(0, r-45.0)^2 + 10*step(x)*x; r=sqrt(x*x+y*y+z*z)")
        system.addForce(wall)
        for ii in range(system.getNumParticles()):
            wall.addParticle(ii)
        bond_stretch.addBond(0, num_beads - 1, 3.0, bond_k)

    elif custom_force == 'e':  # Linked chains
        wall = mm.CustomExternalForce("1*max(0, r-45.0)^2; r=sqrt(x*x+y*y+z*z)")
        system.addForce(wall)
        for ii in range(system.getNumParticles()):
            wall.addParticle(ii)

        # Define topology-based custom force
        topo = mm.CustomNonbondedForce(
            "eps * cos(3.1415926535/(3.3*(1 + step(eps-3.1)*(5)))*r) * (1 - tanh(2*(r - sig))); "
            "sig=(sig1+sig2); eps = min(eps1,eps2)"
        )
        system.addForce(topo)
        topo.addPerParticleParameter("sig")
        topo.addPerParticleParameter("eps")

        # Assign sigma and epsilon values to each particle
        for ii in range(system.getNumParticles()):
            pars = [init_coor[3][6]]  # vdw_sigma remains unchanged
            if ii % 10 in {0, 1, 9}:
                pars.append(5.0)
            else:
                pars.append(30.0)
            topo.addParticle(pars)

        # Add exclusions for neighboring particles
        for ll in range(1, n_skip_neighbors + 1):
            for ii in range(num_beads - ll):
                topo.addExclusion(ii, ii + ll)

        # Add chain linking bonds
        bond_stretch.addBond(0, num_beads // 2 - 1, 3.0, bond_k)
        bond_stretch.addBond(num_beads // 2, num_beads - 1, 3.0, bond_k)

    elif custom_force == 'l':  # Lamina attraction
        wall = mm.CustomExternalForce("1*max(0, r-53.5)^2; r=sqrt(x*x+y*y+z*z)")
        system.addForce(wall)
        for ii in range(system.getNumParticles()):
            wall.addParticle(ii)

        lamin = mm.CustomExternalForce("0.05*(r-52.0)^2; r=sqrt(x*x+y*y+z*z)")
        system.addForce(lamin)
        for ii in range(system.getNumParticles()):
            if resid[ii] > 29:
                lamin.addParticle(ii)


def simulate_polymer():
    """Runs the polymer simulation."""

    radius = (num_beads / polymer_density) ** (1/3)
    print(f"Radius = {radius:8.1f}")

    # Load OpenMM plugins
    mm.Platform.loadPluginsFromDirectory(mm.Platform.getDefaultPluginsDirectory())

    # Initialize system and forces
    system = mm.System()
    nonbond = mm.NonbondedForce()
    bond_stretch = mm.HarmonicBondForce()
    bond_bend = mm.HarmonicAngleForce()

    system.addForce(nonbond)
    if bondStretchStatus:
        system.addForce(bond_stretch)
    if bondBendStatus:
        system.addForce(bond_bend)

    # Read input coordinates
    init_pos_in_nm = []
    with open(inpcrd, "r") as input_coor:
        init_coor = np.loadtxt(input_coor).reshape(num_beads, 12)
    #############################################################    
    
    #init_coor[:, 5] = 0.01
    #init_coor[:, 6] = 28.0 # Temporary
    #init_coor[:, 7] = 0.3

    chain = init_coor[:, 0].astype(int)
    resid = np.ones(num_beads, dtype=int)

    for ii in range(num_beads):
        init_pos_in_nm.append(mm.Vec3(init_coor[ii, 1], init_coor[ii, 2], init_coor[ii, 3]))
        system.addParticle(mass)

    # Read chromatin interactions
    chrom_coor = []
    chrom_e = []
    chrom_type = []
    chrom_parts = 0
    with open(incons, "r") as chrom_file:
        for line in chrom_file:
            fields = line.split()
            chrom_type.append(fields[0])
            chrom_coor.append((int(fields[1]), int(fields[2])))
            chrom_e.append(float(fields[3]))
            chrom_parts += 1

    parse_chrom_info(chrom_type, chrom_coor, chrom_e, init_coor, resid, chrom_parts)

    for ii in range(num_beads):
        nonbond.addParticle(init_coor[ii, 5], 2.0 * init_coor[ii, 6], init_coor[ii, 7])

    if bondStretchStatus:
        for ii in range(num_beads - 1):
            if chain[ii+1] == chain[ii]:
                bond_stretch.addBond(ii, ii+1, init_coor[ii, 4], bond_k)

    if bondBendStatus:
        for kk in range(1, n_cons_angles + 1):
            for ii in range(num_beads - 2 * kk):
                if chain[ii+kk] == chain[ii] and chain[ii+2*kk] == chain[ii]:
                    bond_bend.addAngle(ii, ii+kk, ii+2*kk, init_coor[ii, 8], init_coor[ii, 9])

    if bondStretchStatus:
        for ll in range(1, n_skip_neighbors + 1):
            for ii in range(num_beads - ll):
                nonbond.addException(ii, ii+ll, 0.0, init_coor[ii, 7], 0.0, False)

    if NewInteraction:
        fij = mm.CustomBondForce(CustBndForce)
        system.addForce(fij)
        fij.addPerBondParameter("r0")
        fij.addPerBondParameter("k")

        for ii in range(chrom_parts):
            fij.addBond(chrom_coor[ii][0], chrom_coor[ii][1], [init_coor[3][6], chrom_e[ii]])
    
    apply_custom_force(system, bond_stretch, custom_force, resid, init_coor)
    
    # BEFORE creating the context, assign force groups
    force_group_map = {
        "HarmonicBondForce": 1,
        "HarmonicAngleForce": 2,
        "PeriodicTorsionForce": 3,
        "NonbondedForce": 4,
        "CustomBondForce": 5
    }
    
    # Assign force groups
    for i in range(system.getNumForces()):
        force = system.getForce(i)
        force_name = type(force).__name__
        if force_name in force_group_map:
            force.setForceGroup(force_group_map[force_name])
    
    
    integrator = mm.VariableLangevinIntegrator(temperature, friction, err_tol)
    platform = mm.Platform.getPlatformByName("CUDA")
    context = mm.Context(system, integrator, platform)

    print(f"REMARK  Using OpenMM platform {context.getPlatform().getName()}")
    
    context.setPositions(init_pos_in_nm)

    if minz_steps > 1:
        mm.LocalEnergyMinimizer.minimize(context, minz_tol, minz_steps)

    # Main simulation loop
    for frame_num in range(1, int(total_time_ps / integration_step) + 1):
        state = context.getState(getPositions=True, getEnergy=True)
        
        # Write PDB and RST before advancing simulation
        write_pdb_frame(frame_num, state, PDB_Conversion, state.getTime(), chain, resid, PDB_Precision)
        writerst(state, init_coor)
        
        # Advance simulation step
        integrator.step(integration_step)
    
        # Extract updated energy values AFTER integration step
        state = context.getState(getEnergy=True)
        total_energy, forces = extract_energy(state, system, context, force_group_map)
    
        # Format energy components as a string
        forces_str = " ".join([f"{name}: {energy:.3f}" for name, energy in forces.items()])
    
        # Print everything in one line
        print(f"Total: {total_energy:.3f} {forces_str}", flush=True)
    

        

if __name__ == "__main__":
    try:
        simulate_polymer()
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
