# Parcial-Ciencia-de-Datos

## Instalacion de dependencias

Crear entorno virtual (si aun no existe):

```bash
python3 -m venv .venv
```

Instalar librerias desde requirements:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## Ejecutar la app Streamlit

```bash
.venv/bin/python -m streamlit run ParcialStremlit.py
```

## Despliegue en Streamlit Cloud

- El despliegue usa [requirements.txt](requirements.txt) para instalar dependencias.
- La version de Python para Cloud se fija en [runtime.txt](runtime.txt) (`python-3.11.9`).
