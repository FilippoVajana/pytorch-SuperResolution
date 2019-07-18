#!/bin/bash
clear
echo "Sync project with Rsync"

rsync -rvh --progress -e'ssh -i /mnt/c/Users/filip/.ssh/FilippoVajanaOrobix_rsa' '/mnt/c/Users/filip/Documents/Projects/SuperResolution' filippo_vajana@192.168.4.169:/home/filippo_vajana/Projects/

# -e"ssh -i /mnt/c/Users/filip/.ssh/FilippoVajanaOrobix_rsa"