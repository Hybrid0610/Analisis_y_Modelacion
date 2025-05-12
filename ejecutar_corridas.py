import subprocess
from multiprocessing import Pool
import pandas as pd
import re
import time
from datetime import datetime

def parsear_resultado(output):
    """Extrae métricas de la salida impresa por parallel_BFOA.py"""
    datos = {
        'Fitness': None,
        'BlosumScore': None,
        'Interaction': None,
        'Tiempo': None,
        'NFE': None
    }
    
    # Expresiones regulares para capturar valores (ajustable según los prints)
    patrones = {
        'Fitness': r"Fitness:\s+([\d.]+)",
        'BlosumScore': r"BlosumScore\s+([\d.]+)",
        'Interaction': r"Interaction:\s+([\d.]+)",
        'Tiempo': r"---\s+([\d.]+)\s+seconds",
        'NFE': r"NFE:\s+(\d+)"
    }
    
    for key, patron in patrones.items():
        match = re.search(patron, output)
        if match:
            datos[key] = float(match.group(1)) if key != 'NFE' else int(match.group(1))
    
    return datos

def ejecutar_corrida(corrida_id):
    """Ejecuta una corrida y devuelve métricas estructuradas"""
    print(f"Corrida {corrida_id + 1} iniciada...")
    
    # Ejecuta parallel_BFOA_Modificado.py y captura salida
    comando = ["python", "parallel_BFOA_Modificado.py"]
    proceso = subprocess.run(comando, capture_output=True, text=True)
    
    # Parseo de resultados
    resultados = parsear_resultado(proceso.stdout)
    resultados['Corrida'] = corrida_id + 1
    resultados['Timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"Corrida {corrida_id + 1} completada. Fitness: {resultados['Fitness']}")
    return resultados

if __name__ == "__main__":
    # Configuración
    num_corridas = 30
    procesos_paralelos = 3  # Ajusta según tu CPU
    
    # Ejecución paralela
    with Pool(procesos_paralelos) as pool:
        datos = pool.map(ejecutar_corrida, range(num_corridas))
    
    # Crear DataFrame y guardar Excel
    df = pd.DataFrame(datos)
    
    # Ordenar columnas
    column_order = ['Corrida', 'Fitness', 'BlosumScore', 'Interaction', 'NFE', 'Tiempo', 'Timestamp']
    df = df[column_order]
    
    # Guardar en Excel (con hoja de parámetros)
    with pd.ExcelWriter("resultados_corridas_mod.xlsx", engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Resultados', index=False)
        
        # Hoja adicional con parámetros usados
        parametros = {
            'Bacterias': [10],
            'Iteraciones': [10],
            'Tumbo': [2],
            'dAttr': [0.2],
            'wAttr': [0.002],
            'wRep': [0.001]
        }
        pd.DataFrame(parametros).to_excel(writer, sheet_name='Parametros', index=False)
    
    print("\nExcel generado: 'resultados_corridas_mod.xlsx'")
    print("Resumen estadístico:")
    print(df[['Fitness', 'BlosumScore', 'Tiempo']].describe())