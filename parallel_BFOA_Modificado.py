from copy import deepcopy
from multiprocessing import Manager
import time
import numpy as np
from bacteria import bacteria
from fastaReader import fastaReader


def calcular_tumbo_adaptativo(iteracion_actual, iteraciones_totales, tumbo_inicial, fitness_max):
    """Tumbo adaptativo"""
    enfriamiento = 0.5 ** (iteracion_actual / iteraciones_totales)
    reduccion_anomalia = 1.0
    if fitness_max > 500:  # Umbral ajustado
        reduccion_anomalia = max(0.01, 500 / fitness_max)  # Limita reducción mínima al 1%
    
    tumbo_actual = tumbo_inicial * enfriamiento * reduccion_anomalia
    return max(min(tumbo_actual, tumbo_inicial), 5)  # Asegura 5 ≤ tumbo ≤ tumbo_inicial

def corregir_anomalia(poblacion, idx, secuencias, veryBest, operadorBacterial):
    """
    Corrige una bacteria con fitness anómalo y devuelve:
    - tuple: (éxito: bool, nuevo_fitness: float)
    """
    try:
        # 1. Preparar secuencia de respaldo (asegurando que sea lista)
        if veryBest[0] is not None and veryBest[1] < 100:  # Usar veryBest si es válido
            secuencia_respaldo = deepcopy(veryBest[2])
        else:  # Usar secuencia original
            secuencia_respaldo = deepcopy(secuencias[idx % len(secuencias)])
        
        if isinstance(secuencia_respaldo, str):
            secuencia_respaldo = list(secuencia_respaldo)
        
        # 2. Aplicar corrección
        poblacion[idx] = secuencia_respaldo
        
        # 3. Recalcular fitness específico para esta bacteria
        operadorBacterial.creaGranListaPares([poblacion[idx]])
        operadorBacterial.evaluaBlosum()
        nuevo_fitness = operadorBacterial.tablaFitness[0]
        
        # 4. Validar que la corrección fue efectiva
        if nuevo_fitness > 100:  # Si persiste el problema
            raise ValueError("Fitness corregido sigue siendo anómalo")
            
        return (True, nuevo_fitness)
        
    except Exception as e:
        print(f"Error en corrección: {str(e)}")
        # Secuencia mínima de respaldo
        poblacion[idx] = ['A','T','G','C']  # Secuencia de ADN mínima
        return (False, 0)

if __name__ == "__main__":
    # Parámetros iniciales
    numeroDeBacterias = 6
    iteraciones = 5
    tumbo_inicial = 300
    nado = 3
    
    # Parámetros de atracción/repulsión ajustados
    dAttr = 0.15
    wAttr = 0.005 
    hRep = dAttr
    wRep = 0.003  
    
    # Lectura de secuencias
    # Asegura que todas las secuencias sean listas
    secuencias = [list(seq) if isinstance(seq, str) else seq for seq in fastaReader().seqs]
    names = fastaReader().names
    
    #hace todas las secuencias listas de caracteres
    for i in range(len(secuencias)):
        #elimina saltos de linea
        secuencias[i] = list(secuencias[i])
        
        
    
    numSec = len(secuencias)
    print(f"Número de secuencias: {numSec}")

    manager = Manager()
    poblacion = manager.list([deepcopy(secuencias) for _ in range(numeroDeBacterias)])
    
    operadorBacterial = bacteria(numeroDeBacterias)
    veryBest = [None, -np.inf, None]
    globalNFE = 0
    start_time = time.time()

    # En la inicialización, verifica todas las bacterias
    for i in range(len(poblacion)):
        if not poblacion[i]:  # Si está vacía
            poblacion[i] = deepcopy(secuencias[i % len(secuencias)])
        

    for it in range(iteraciones):
        print(f"\n--- Iteración {it+1}/{iteraciones} ---")
        
        # 1. Cálculo de tumbo adaptativo
        current_max_fitness = veryBest[1] if veryBest[0] is not None else 0
        tumbo_actual = calcular_tumbo_adaptativo(it, iteraciones, tumbo_inicial, current_max_fitness)
        tumbo_int = int(tumbo_actual)
        print(f"Tumbo actual: {tumbo_int:.2f}")
        
        # 2. Movimiento bacteriano controlado
        operadorBacterial.tumbo(numSec, poblacion, tumbo_int)
        operadorBacterial.cuadra(numSec, poblacion)
        
        # 3. Evaluación con controles
        try:
            operadorBacterial.creaGranListaPares(poblacion)
            operadorBacterial.evaluaBlosum()
            
            # Versión segura de creación de tablas
            operadorBacterial.creaTablasAtractRepel(poblacion, dAttr, wAttr, hRep, wRep)
            operadorBacterial.creaTablaInteraction()
            
            # Función de fitness balanceada
            operadorBacterial.creaTablaFitness()
            
            # Validación de resultados
            if not hasattr(operadorBacterial, 'tablaFitness') or any(f is None for f in operadorBacterial.tablaFitness):
                raise ValueError("Fitness no calculado correctamente")
                
        except Exception as e:
            print(f"Error en evaluación: {str(e)}")
            print("Reiniciando población...")
            poblacion = manager.list([deepcopy(secuencias) for _ in range(numeroDeBacterias)])
            continue
        
        # 4. Selección con verificación
        
        globalNFE += operadorBacterial.getNFE()
        bestIdx, bestFitness = operadorBacterial.obtieneBest(globalNFE)
        
        # Dentro del bucle principal, después de obtener bestFitness:

        UMBRAL_FITNESS = 100  # Ajustar según necesidades

        if bestFitness > UMBRAL_FITNESS:
            print(f"\n¡ANOMALÍA DETECTADA! Fitness: {bestFitness}")
            print(f"Corrigiendo bacteria {bestIdx}...")
            
            # Paso 1: Corregir la bacteria problemática
            exito, nuevo_fitness = corregir_anomalia(poblacion, bestIdx, secuencias, veryBest, operadorBacterial)
            
            # Paso 2: Actualizar todas las estructuras de datos
            if exito:
                print(f"Corrección exitosa. Nuevo fitness: {nuevo_fitness}")
                
                # Actualizar tablaFitness completa
                operadorBacterial.creaGranListaPares(poblacion)
                operadorBacterial.evaluaBlosum()
                operadorBacterial.creaTablaFitness()
                
                # Forzar actualización de bestFitness y veryBest
                bestFitness = nuevo_fitness
                if veryBest[1] > UMBRAL_FITNESS or nuevo_fitness > veryBest[1]:
                    veryBest = [bestIdx, nuevo_fitness, deepcopy(poblacion[bestIdx])]
            else:
                print("¡Corrección falló! Descartando bacteria.")
                bestFitness = 0
                poblacion[bestIdx] = deepcopy(secuencias[bestIdx % len(secuencias)])  # Reset completo

            # Debug detallado
            print("\nEstado post-corrección:")
            print(f"Bacteria {bestIdx}: {operadorBacterial.tablaFitness[bestIdx]}")
            if bestIdx > 0:
                print(f"Bacteria {bestIdx-1}: {operadorBacterial.tablaFitness[bestIdx-1]}")
            if bestIdx < len(poblacion)-1:
                print(f"Bacteria {bestIdx+1}: {operadorBacterial.tablaFitness[bestIdx+1]}")
                
                if bestIdx is not None and bestFitness is not None:
                    print(f"Mejor fitness actual: {bestFitness:.2f}")
                    if bestFitness > veryBest[1]:
                        veryBest = [bestIdx, bestFitness, deepcopy(poblacion[bestIdx])]
                        print(f"Nuevo mejor global: {bestFitness:.2f}")
                        
                if operadorBacterial.tablaFitness[bestIdx] > UMBRAL_FITNESS:
                    print("¡Persiste anomalia! Aplicando corrección final...")
                    bestFitness = corregir_anomalia(poblacion, bestIdx, secuencias, veryBest, operadorBacterial)
                    operadorBacterial.tablaFitness[bestIdx] = bestFitness  # Actualiza tabla
        
        # 5. Reemplazo seguro
        if veryBest[0] is not None:
            operadorBacterial.replaceWorst(poblacion, veryBest[0])
        
        operadorBacterial.resetListas(numeroDeBacterias)

    # Resultados finales
    print("\n--- Resultados Finales ---")
    if veryBest[0] is not None:
        print(f"Mejor fitness: {veryBest[1]:.2f}")
    else:
        print("No se encontraron soluciones válidas")
    print(f"Tiempo total: {time.time() - start_time:.2f} segundos")
