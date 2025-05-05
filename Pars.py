total_time_ps      = 5e7 #
integration_step   = 1e5 #

minz_steps         = 5000  #5e6
minz_tol           = 1.0

friction           = 0.003 # 1/ps
err_tol            = 0.003
    
mass               = 100.0
temperature        = 300.0

polymer_density    = 0.05 # Ignore
    
bond_k             = 100.0

num_beads          = 50000
custom_force       = '-' #'e'
CustBndForce       = "0.5*k*(r-(2*r0))^2" # Forced looping

bondStretchStatus  = 1   # 1: Add bond
bondBendStatus     = 1   # 1: Add angle
NewInteraction     = 1   # 1: Add new interaction

n_skip_neighbors   = 3   # 3 by default
n_cons_angles      = 1
PDB_Precision      = 2   # Use 3 by default
PDB_Conversion     = 0.1 # Use 1 by default: Note that you are getting in units of 0.1 nm if set to 0.1

