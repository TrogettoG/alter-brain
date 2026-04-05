#!/bin/bash
cd "/Users/trogettog/Documents/Proyects/Experimentos IA/alter_brain"

# Cargar variables de entorno desde .env
set -a
source .env
set +a

# Ejecutar con el Python que tiene todos los paquetes instalados
exec /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 alter_daemon.py
