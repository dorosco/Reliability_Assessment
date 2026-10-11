import pandas as pd
import pulp
import sys
import time
start_time = time.time()
# Redirigir toda la salida al archivo
sys.stdout = open("salida_MULTI_ALIMENTADOR.txt", "w", encoding='utf-8')
# Leemos los datos desde archivos CSV
DATOS_LINEA = pd.read_csv('DATA_LINEAS.csv')  
DATOS_NODOS = pd.read_csv('DATA_NODOS.csv')   
# CONJUNTOS BÁSICOS
PSI_N = set(DATOS_LINEA['u'].unique()) | set(DATOS_LINEA['v'].unique())  # Todos los nodos
PSI_LN_TEMP = set(DATOS_NODOS['Node'].unique())  # Nodos de carga
PSI_N_LIST = list(PSI_N)
UPSILON = [(int(row['u']), int(row['v'])) for _, row in DATOS_LINEA.iterrows()]
# IDENTIFICAR SUBESTACIONES (nodos que no están en PSI_LN_TEMP)
PSI_SS_LIST = list(PSI_N - PSI_LN_TEMP)  # Múltiples subestaciones
PSI_LN = list(PSI_LN_TEMP)
print(f"="*500)
print(f"Nodos totales: {sorted(PSI_N_LIST)}")
print(f"Nodos de carga: {sorted(PSI_LN_TEMP)}")
print(f"Subestaciones identificadas: {sorted(PSI_SS_LIST)}")
print(f"Número de alimentadores: {len(PSI_SS_LIST)}")

# IDENTIFICAR ALIMENTADORES (cada subestación define un alimentador)
def identificar_alimentadores():

    alimentadores = {}
    nodos_visitados = set()
    
    # Crear grafo de adyacencia SIN defaultdict
    grafo = {}
    for u, v in UPSILON:
        # Inicializar listas si no existen
        if u not in grafo:
            grafo[u] = []
        if v not in grafo:
            grafo[v] = []
        
        # Agregar conexiones
        grafo[u].append(v)
        grafo[v].append(u)
    
    for ss in PSI_SS_LIST:
        alimentadores[ss] = []
        cola = [ss]
        visitados_local = {ss}
        
        while cola:
            nodo_actual = cola.pop(0)
            
            # Verificar si el nodo existe en el grafo
            if nodo_actual in grafo:
                for vecino in grafo[nodo_actual]:
                    if vecino not in visitados_local:
                        visitados_local.add(vecino)
                        if vecino in PSI_LN_TEMP:  # Solo agregar nodos de carga
                            alimentadores[ss].append(vecino)
                        cola.append(vecino)
    
    return alimentadores

ALIMENTADORES = identificar_alimentadores()

print(f"="*500)
print("\nAlimentadores identificados:")
for ss, nodos in ALIMENTADORES.items():
    print(f"  Subestación {ss}: {sorted(nodos)} ({len(nodos)} nodos)")

# CONJUNTOS ADYACENTES
PSI_ADY = {}
for nodo in PSI_N_LIST:
    PSI_ADY[nodo] = []
for u, v in UPSILON:
    PSI_ADY[u].append(v)
    PSI_ADY[v].append(u)

LAMBDA_DICT = {}
TAU_RP_DICT = {}
TAU_SW_DICT = {}
for _, row in DATOS_LINEA.iterrows():
    u, v = int(row['u']), int(row['v'])
    for (i, j) in [(u, v), (v, u)]:
        LAMBDA_DICT[(i, j)] = row['lambda']
        TAU_RP_DICT[(i, j)] = row['Tao_RP']
        TAU_SW_DICT[(i, j)] = row['Tao_SW']

# Diccionarios para datos de nodos
N_C_DICT = {}
L_MW_DICT = {}
for _, row in DATOS_NODOS.iterrows():
    nodo = int(row['Node'])
    N_C_DICT[nodo] = row['NC']
    L_MW_DICT[nodo] = row['L_MW']

print("\n" + "="*100)
print("RESOLVIENDO PROBLEMAS DE FLUJO POR ALIMENTADOR")
print("="*100)

# Resolver problemas de flujo para cada alimentador por separado
resultados_por_alimentador = {}
f_tilde = {}  # Soluciones óptimas por alimentador
f_hat = {}    # Soluciones óptimas por alimentador

for ss in PSI_SS_LIST:
    print(f"\n--- Procesando Alimentador de Subestación {ss} ---")
    
    # Nodos del alimentador actual
    nodos_alimentador = [ss] + ALIMENTADORES[ss]
    print(f"Nodos del alimentador: {sorted(nodos_alimentador)}")
    
    # Líneas del alimentador actual
    lineas_alimentador = []
    for u, v in UPSILON:
        if u in nodos_alimentador and v in nodos_alimentador:
            lineas_alimentador.append((u, v))
    
    print(f"Líneas del alimentador: {lineas_alimentador}")
    
    # Inicializar estructuras para este alimentador
    f_tilde[ss] = {}
    f_hat[ss] = {}
    resultados_por_alimentador[ss] = []
    
    # Resolver para cada nodo de carga en este alimentador
    for s in ALIMENTADORES[ss]:
        print(f"  Resolviendo para nodo {s}")
        
        # Crear problema de optimización
        prob = pulp.LpProblem(f"Flujo_Alimentador_{ss}_Nodo_{s}", pulp.LpMinimize)
        
        # Variables de decisión solo para nodos del alimentador
        f_vars = {}
        g_vars = {}
        
        # Variables f_ijs solo para líneas del alimentador
        for i, j in lineas_alimentador:
            f_vars[(i, j, s)] = pulp.LpVariable(f"f_{i}{j}{s}", 0, 1)
            f_vars[(j, i, s)] = pulp.LpVariable(f"f_{j}{i}{s}", 0, 1)
        
        # Variable de generación solo para la subestación del alimentador
        g_vars[(ss, s)] = pulp.LpVariable(f"g_{ss}{s}", 0, 1)
        
        # Función objetivo
        objetivo = [var for var in f_vars.values()]
        prob += pulp.lpSum(objetivo)
        
        # Restricciones para cada nodo del alimentador
        for i in nodos_alimentador:
            flujo_neto = []
            
            # Solo considerar vecinos dentro del alimentador
            for j in PSI_ADY[i]:
                if j in nodos_alimentador:
                    if (i, j, s) in f_vars and (j, i, s) in f_vars:
                        flujo_neto.append(f_vars[(i, j, s)])
                        flujo_neto.append(-f_vars[(j, i, s)])
            
            # Definir L_is y g_is
            L_is = 1 if i == s else 0
            g_is = g_vars[(ss, s)] if i == ss else 0
            
            # Restricción de balance
            if flujo_neto:
                prob += pulp.lpSum(flujo_neto) == g_is - L_is
            else:
                prob += 0 == g_is - L_is
        
        # Resolver
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Almacenar soluciones
        if prob.status == pulp.LpStatusOptimal:
            if s not in f_tilde:
                f_tilde[s] = {}
                f_hat[s] = {}
            
            for key, var in f_vars.items():
                valor = pulp.value(var)
                i, j, nodo_s = key
                
                f_tilde[s][(i, j)] = valor if valor > 1e-6 else 0
                f_hat[s][(i, j)] = valor if valor > 1e-6 else 0
                
                if valor > 1e-6:
                    resultados_por_alimentador[ss].append({
                        'Alimentador': ss,
                        'Nodo_s': nodo_s,
                        'Variable': f'f_{i}{j}{nodo_s}',
                        'Valor': round(valor, 6)
                    })
        else:
            print(f"    ERROR: No se pudo resolver para nodo {s}")

print("\n" + "="*100)
print("CALCULANDO ÍNDICES DE CONFIABILIDAD POR ALIMENTADOR")
print("="*100)

def calcular_indices_confiabilidad_multi():
    """
    Calcula índices de confiabilidad para sistema multi-alimentador
    """
    indices_por_alimentador = {}
    
    for ss in PSI_SS_LIST:
        print(f"\n--- Índices para Alimentador {ss} ---")
        
        # Calcular N_ij^CB para este alimentador
        N_CB = {}
        for i in [ss]:  # Subestación del alimentador
            for j in PSI_ADY[i]:
                if j in ALIMENTADORES[ss]:  # Solo vecinos del mismo alimentador
                    N_CB[(i, j)] = 0
                    for s_prime in ALIMENTADORES[ss]:
                        suma_interna = 0
                        for r in PSI_ADY[s_prime]:
                            if r in [ss] + ALIMENTADORES[ss]:  # Solo dentro del alimentador
                                f_rss = f_tilde[s_prime].get((r, s_prime), 0)
                                lambda_rs = LAMBDA_DICT.get((r, s_prime), 0)
                                suma_interna += f_rss * lambda_rs
                        
                        f_ijs = f_tilde[s_prime].get((i, j), 0)
                        N_CB[(i, j)] += suma_interna * f_ijs
        
        # Calcular D_ij^CB,SW para este alimentador
        D_CB_SW = {}
        for i in [ss]:
            for j in PSI_ADY[i]:
                if j in ALIMENTADORES[ss]:
                    D_CB_SW[(i, j)] = 0
                    for s_prime in ALIMENTADORES[ss]:
                        suma_interna = 0
                        for r in PSI_ADY[s_prime]:
                            if r in [ss] + ALIMENTADORES[ss]:
                                f_rss = f_hat[s_prime].get((r, s_prime), 0)
                                lambda_rs = LAMBDA_DICT.get((r, s_prime), 0)
                                tau_SW_rs = TAU_SW_DICT.get((r, s_prime), 0)
                                suma_interna += f_rss * lambda_rs * tau_SW_rs
                        
                        f_ijs = f_hat[s_prime].get((i, j), 0)
                        D_CB_SW[(i, j)] += suma_interna * f_ijs
        
        # Calcular índices para cada nodo del alimentador
        indices_alimentador = []
        
        for s in ALIMENTADORES[ss]:
            # N_s^RP
            N_s_RP = 0
            for i, j in UPSILON:
                if i in [ss] + ALIMENTADORES[ss] and j in [ss] + ALIMENTADORES[ss]:
                    f_ijs_tilde = f_tilde[s].get((i, j), 0)
                    f_jis_tilde = f_tilde[s].get((j, i), 0)
                    lambda_ij = LAMBDA_DICT[(i, j)]
                    N_s_RP += (f_ijs_tilde + f_jis_tilde) * lambda_ij
            
            # N_s^T
            N_s_T = 0
            for i in [ss]:
                for j in PSI_ADY[i]:
                    if j in ALIMENTADORES[ss] and (i, j) in N_CB:
                        f_ijs_tilde = f_tilde[s].get((i, j), 0)
                        f_jis_tilde = f_tilde[s].get((j, i), 0)
                        N_s_T += N_CB[(i, j)] * (f_ijs_tilde + f_jis_tilde)
            
            # N_s^SW
            N_s_SW = N_s_T - N_s_RP
            
            # D_s^RP
            D_s_RP = 0
            for i, j in UPSILON:
                if i in [ss] + ALIMENTADORES[ss] and j in [ss] + ALIMENTADORES[ss]:
                    f_ijs_tilde = f_tilde[s].get((i, j), 0)
                    f_jis_tilde = f_tilde[s].get((j, i), 0)
                    lambda_ij = LAMBDA_DICT[(i, j)]
                    tau_RP_ij = TAU_RP_DICT[(i, j)]
                    D_s_RP += (f_ijs_tilde + f_jis_tilde) * lambda_ij * tau_RP_ij
            
            # D_s^T,SW
            D_s_T_SW = 0
            for i in [ss]:
                for j in PSI_ADY[i]:
                    if j in ALIMENTADORES[ss] and (i, j) in D_CB_SW:
                        f_ijs_hat = f_hat[s].get((i, j), 0)
                        f_jis_hat = f_hat[s].get((j, i), 0)
                        D_s_T_SW += D_CB_SW[(i, j)] * (f_ijs_hat + f_jis_hat)
            
            # D̄_s^SW
            D_bar_s_SW = 0
            for i, j in UPSILON:
                if i in [ss] + ALIMENTADORES[ss] and j in [ss] + ALIMENTADORES[ss]:
                    f_ijs_hat = f_hat[s].get((i, j), 0)
                    f_jis_hat = f_hat[s].get((j, i), 0)
                    lambda_ij = LAMBDA_DICT[(i, j)]
                    tau_SW_ij = TAU_SW_DICT[(i, j)]
                    D_bar_s_SW += (f_ijs_hat + f_jis_hat) * lambda_ij * tau_SW_ij
            
            # D_s^SW
            D_s_SW = D_s_T_SW - D_bar_s_SW
            
            # CIF_s y CID_s
            CIF_s = N_s_RP + N_s_SW
            CID_s = D_s_RP + D_s_SW
            
            indices_alimentador.append({
                'Alimentador': ss,
                'Nodo': s,
                'N_s_RP': round(N_s_RP, 6),
                'N_s_SW': round(N_s_SW, 6),
                'D_s_RP': round(D_s_RP, 6),
                'D_s_SW': round(D_s_SW, 6),
                'CIF_s': round(CIF_s, 6),
                'CID_s': round(CID_s, 6)
            })
        
        indices_por_alimentador[ss] = indices_alimentador
    
    return indices_por_alimentador

def calcular_indices_sistema_multi(indices_por_alimentador):
    """
    Calcula índices del sistema completo (todos los alimentadores)
    """
    # Consolidar todos los índices
    todos_indices = []
    for ss in PSI_SS_LIST:
        todos_indices.extend(indices_por_alimentador[ss])
    
    # Calcular totales
    total_clientes = sum(N_C_DICT[s] for s in PSI_LN)
    
    # SAIFI del sistema
    numerador_saifi = sum(N_C_DICT[item['Nodo']] * item['CIF_s'] for item in todos_indices)
    SAIFI = numerador_saifi / total_clientes
    
    # SAIDI del sistema
    numerador_saidi = sum(N_C_DICT[item['Nodo']] * item['CID_s'] for item in todos_indices)
    SAIDI = numerador_saidi / total_clientes
    
    # ASAI del sistema
    ASAI = 1 - (SAIDI / 8760)
    ASAI_percent = ASAI * 100
    
    # EENS del sistema
    EENS = sum(item['CID_s'] * L_MW_DICT[item['Nodo']] for item in todos_indices)
    
    return {
        'SAIFI': round(SAIFI, 6),
        'SAIDI': round(SAIDI, 6),
        'ASAI': round(ASAI_percent, 6),
        'EENS': round(EENS, 6),
        'total_clientes': total_clientes,
        'total_nodos': len(PSI_LN),
        'total_alimentadores': len(PSI_SS_LIST)
    }

# Ejecutar cálculos
indices_multi = calcular_indices_confiabilidad_multi()
indices_sistema_multi = calcular_indices_sistema_multi(indices_multi)

# PRESENTACIÓN DE RESULTADOS
print("\n" + "="*120)
print("RESULTADOS POR ALIMENTADOR")
print("="*120)

for ss in PSI_SS_LIST:
    print(f"\n--- ALIMENTADOR {ss} ---")
    print(f"{'Nodo':<6} {'N_s^RP':<10} {'N_s^SW':<10} {'D_s^RP':<10} {'D_s^SW':<10} {'CIF_s':<10} {'CID_s':<10}")
    print("-" * 70)
    
    for item in indices_multi[ss]:
        print(f"{item['Nodo']:<6} {item['N_s_RP']:<10} {item['N_s_SW']:<10} "
              f"{item['D_s_RP']:<10} {item['D_s_SW']:<10} {item['CIF_s']:<10} {item['CID_s']:<10}")

print("\n" + "="*120)
print("RESUMEN CONSOLIDADO POR NODO")
print("="*120)
print(f"{'Alim':<6} {'Nodo':<6} {'CIF_s':<15} {'CID_s':<15} {'Clientes':<10} {'Carga (MW)':<12}")
print("-" * 70)

for ss in PSI_SS_LIST:
    for item in indices_multi[ss]:
        nodo = item['Nodo']
        print(f"{ss:<6} {nodo:<6} {item['CIF_s']:<15} {item['CID_s']:<15} "
              f"{N_C_DICT[nodo]:<10} {L_MW_DICT[nodo]:<12}")

print("\n" + "="*120)
print("ÍNDICES DE CONFIABILIDAD DEL SISTEMA COMPLETO")
print("="*120)
print(f"{'Parámetro':<30} {'Valor':<15} {'Unidad':<20}")
print("-" * 70)
print(f"{'Número de alimentadores':<30} {indices_sistema_multi['total_alimentadores']:<15} {'alimentadores':<20}")
print(f"{'Número de nodos':<30} {indices_sistema_multi['total_nodos']:<15} {'nodos':<20}")
print(f"{'Número de clientes':<30} {indices_sistema_multi['total_clientes']:<15} {'clientes':<20}")
print(f"{'SAIFI':<30} {indices_sistema_multi['SAIFI']:<15} {'interrupciones/año':<20}")
print(f"{'SAIDI':<30} {indices_sistema_multi['SAIDI']:<15} {'h/año':<20}")
print(f"{'ASAI':<30} {indices_sistema_multi['ASAI']:<15} {'%':<20}")
print(f"{'EENS':<30} {indices_sistema_multi['EENS']:<15} {'MWh/año':<20}")

print("\n" + "="*120)
print("ÍNDICES POR ALIMENTADOR")
print("="*120)

for ss in PSI_SS_LIST:
    # Calcular índices específicos del alimentador
    nodos_alim = [item for item in indices_multi[ss]]
    clientes_alim = sum(N_C_DICT[item['Nodo']] for item in nodos_alim)
    
    saifi_alim = sum(N_C_DICT[item['Nodo']] * item['CIF_s'] for item in nodos_alim) / clientes_alim
    saidi_alim = sum(N_C_DICT[item['Nodo']] * item['CID_s'] for item in nodos_alim) / clientes_alim
    asai_alim = (1 - saidi_alim / 8760) * 100
    eens_alim = sum(item['CID_s'] * L_MW_DICT[item['Nodo']] for item in nodos_alim)
    
    print(f"\nAlimentador {ss}:")
    print(f"  Nodos: {len(nodos_alim)}")
    print(f"  Clientes: {clientes_alim}")
    print(f"  SAIFI: {round(saifi_alim, 6)} int/año")
    print(f"  SAIDI: {round(saidi_alim, 6)} h/año")
    print(f"  ASAI: {round(asai_alim, 6)} %")
    print(f"  EENS: {round(eens_alim, 6)} MWh/año")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTiempo total de resolución: {elapsed_time:.4f} segundos")

# Cerrar archivo de salida
sys.stdout.close()
sys.stdout = sys.__stdout__

print("Análisis multi-alimentador completado. Resultados guardados en 'salida_MULTI_ALIMENTADOR.txt'")