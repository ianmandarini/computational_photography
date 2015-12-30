#================================================================================
#
# University of California, Berkeley
# CS194-26 (CS294-26): Computational Photography"
#
#================================================================================
#
# Project 4: "Seam Carving"
#
# Student: Ian Albuquerque Raymundo da Silva"
# Email: ian.albuquerque@berkeley.edu
#
#================================================================================
#
# Special thanks to Alexei Alyosha Efros, Rachel Albert and Weilun Sun for help
# during lectures, office hours and questions on Piazza.
#
#================================================================================

#================================================================================
# File Description: 
#
# main.py
#
# Asks the user if they want to use the precompute function or the online
# function.
#
# Acts as the user interface.
#================================================================================

# The two main modules of the assignment:

# precompute.py
import precompute

# online.py
import online

#----------------------------------------------------------------------

# CHANGE THIS ONLY IF YOU WANT TO USE A MASK FOR THE ENERGY FUNCTION
PRE_USE_ENERGY_MASK = False

PRE_INPUT_FILE_NAME = "botinas.jpg"
PRE_OUTPUT_FILE_NAME = "botinas.ord"
PRE_ENERGY_MASK = "botinas_mask.jpg"

ON_INPUT_FILE_NAME = PRE_INPUT_FILE_NAME
ON_INPUT_ORDER = PRE_OUTPUT_FILE_NAME

#----------------------------------------------------------------------

if __name__ == '__main__':
	option = raw_input("Precompute (p) or online (o)? ")
	if option == "p":
		precompute.run(PRE_INPUT_FILE_NAME, PRE_OUTPUT_FILE_NAME, PRE_USE_ENERGY_MASK, PRE_ENERGY_MASK)
	else:
		online.run(ON_INPUT_FILE_NAME, ON_INPUT_ORDER)