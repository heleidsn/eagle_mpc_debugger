#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make the Python script executable
chmod +x "$SCRIPT_DIR/../eagle_mpc_gui.py"

# Run the GUI
rosrun eagle_mpc_debugger eagle_mpc_gui.py 