#!/usr/bin/env bash

# Unpack batch.tar.gz
tar zxf archives/batch.tar.gz

# Unpack batch_xyz.tar.gz
tar zxf archives/batch_xyz.tar.gz

# Gather and unpack xyz.tar.gz
cat archives/xyz?? | tar zx

# Gather and unpack log_files.tar.gz
cat archives/log_files?? | tar zx
