import pandas as pd
import networkx as nx
from collections import defaultdict
import os

class RadialNetworkReliability:
    def __init__(self, data_file_path="DATA.csv", customer_data_file="6_NODOS.csv"):
        """
        Análisis de confiabilidad para redes radiales - Versión optimizada
        Carga datos desde DATA.csv y calcula índices de confiabilidad
        """
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {data_file_path}")
        
        try:
            # Cargar datos del CSV
            self.df = pd.read_csv(data_file_path)
            required_columns = ['u', 'v', 'lambda', 'Tao_RP', 'Tao_SW']
            for col in required_columns:
                if col not in self.df.columns:
                    raise ValueError(f"Columna requerida faltante: {col}")
            
            print(f"=== ANÁLISIS DE CONFIABILIDAD DE RED RADIAL ===")
            print(f"Archivo: {data_file_path}")
            print(f"Ramas cargadas: {len(self.df)}")
            
            # Construir red y calcular índices
            self.build_network() 
            self.flow_vars = self.compute_flow_variables()
            self.reliability_indices = self.calculate_all_indices()
            
            # Cargar datos de clientes y calcular índices adicionales
            self.load_customer_data(customer_data_file)
            self.calculate_additional_indices()
            
        except Exception as e:
            print(f"Error: {str(e)}")
            raise
    
    def build_network(self):
        """Construir topología de la red radial"""
        self.G = nx.Graph()
        
        # Agregar aristas con atributos
        for _, row in self.df.iterrows():
            self.G.add_edge(row['u'], row['v'], 
                          lambda_val=row['lambda'],
                          tau_RP=row['Tao_RP'],
                          tau_SW=row['Tao_SW'])
        
        # Identificar nodo raíz y nodos de carga
        self.root = self.find_root_node()
        self.load_nodes = sorted([n for n in self.G.nodes() if n != self.root])
        
        # Calcular caminos desde raíz a cada nodo
        self.paths = {}
        for node in self.load_nodes:
            self.paths[node] = nx.shortest_path(self.G, self.root, node)
    
    def find_root_node(self):
        """Identificar nodo raíz (subestación) - típicamente el de menor grado o menor número"""
        degrees = dict(self.G.degree())
        candidates = [n for n, deg in degrees.items() if deg == 1]
        return candidates[0] if len(candidates) == 1 else min(self.G.nodes())
    
    def compute_flow_variables(self):
        """Calcular variables de flujo f_ijs"""
        flows = defaultdict(int)
        
        for s in self.load_nodes:
            path = self.paths[s]
            for u, v in self.G.edges():
                # Verificar si la arista está en el camino
                for i in range(len(path) - 1):
                    if (path[i] == u and path[i + 1] == v):
                        flows[(u, v, s)] = 1
                    elif (path[i] == v and path[i + 1] == u):
                        flows[(v, u, s)] = 1
        
        return flows
    
    def calculate_N_RP(self, node):
        """Calcular N_s^RP (Ecuación 15)"""
        N_RP = 0.0
        for u, v in self.G.edges():
            f_uvs = self.flow_vars.get((u, v, node), 0)
            f_vus = self.flow_vars.get((v, u, node), 0)
            lambda_uv = self.G[u][v]['lambda_val']
            N_RP += (f_uvs + f_vus) * lambda_uv
        return N_RP
    
    def calculate_N_CB(self):
        """Calcular N_ij^CB (Ecuación 19)"""
        N_CB = {}
        circuit_breakers = [(u, v) for u, v in self.G.edges() 
                           if u == self.root or v == self.root]
        
        for i, j in circuit_breakers:
            N_ij_CB = 0.0
            for s in self.load_nodes:
                f_ijs = self.flow_vars.get((i, j, s), 0)
                f_jis = self.flow_vars.get((j, i, s), 0)
                f_total = f_ijs + f_jis
                
                if f_total > 0:
                    sum_lambda = 0.0
                    for r, sp in self.G.edges():
                        if sp == s and self.flow_vars.get((r, sp, s), 0) == 1:
                            sum_lambda += self.G[r][sp]['lambda_val']
                        elif r == s and self.flow_vars.get((sp, r, s), 0) == 1:
                            sum_lambda += self.G[sp][r]['lambda_val']
                    
                    N_ij_CB += sum_lambda * f_total
            
            N_CB[(i, j)] = N_ij_CB
        
        return N_CB
    
    def calculate_N_T(self, node):
        """Calcular N_s^T (Ecuación 18)"""
        N_CB = self.calculate_N_CB()
        N_T = 0.0
        
        for (i, j), n_cb in N_CB.items():
            f_ijs = self.flow_vars.get((i, j, node), 0)
            f_jis = self.flow_vars.get((j, i, node), 0)
            N_T += n_cb * (f_ijs + f_jis)
        
        return N_T
    
    def calculate_N_SW(self, node):
        """Calcular N_s^SW (Ecuación 17)"""
        return self.calculate_N_T(node) - self.calculate_N_RP(node)
    
    def calculate_D_RP(self, node):
        """Calcular D_s^RP (Ecuación 25)"""
        D_RP = 0.0
        for u, v in self.G.edges():
            f_uvs = self.flow_vars.get((u, v, node), 0)
            f_vus = self.flow_vars.get((v, u, node), 0)
            lambda_uv = self.G[u][v]['lambda_val']
            tau_RP = self.G[u][v]['tau_RP']
            D_RP += (f_uvs + f_vus) * lambda_uv * tau_RP
        return D_RP
    
    def calculate_D_CB_SW(self):
        """Calcular D_ij^CB,SW (Ecuación 30)"""
        D_CB_SW = {}
        circuit_breakers = [(u, v) for u, v in self.G.edges() 
                           if u == self.root or v == self.root]
        
        for i, j in circuit_breakers:
            D_ij_CB_SW = 0.0
            for s in self.load_nodes:
                f_ijs = self.flow_vars.get((i, j, s), 0)
                f_jis = self.flow_vars.get((j, i, s), 0)
                f_total = f_ijs + f_jis
                
                if f_total > 0:
                    sum_lambda_tau = 0.0
                    for r, sp in self.G.edges():
                        if sp == s and self.flow_vars.get((r, sp, s), 0) == 1:
                            lambda_rs = self.G[r][sp]['lambda_val']
                            tau_SW = self.G[r][sp]['tau_SW']
                            sum_lambda_tau += lambda_rs * tau_SW
                        elif r == s and self.flow_vars.get((sp, r, s), 0) == 1:
                            lambda_sr = self.G[sp][r]['lambda_val']
                            tau_SW = self.G[sp][r]['tau_SW']
                            sum_lambda_tau += lambda_sr * tau_SW
                    
                    D_ij_CB_SW += sum_lambda_tau * f_total
            
            D_CB_SW[(i, j)] = D_ij_CB_SW
        
        return D_CB_SW
    
    def calculate_D_T_SW(self, node):
        """Calcular D_s^T,SW (Ecuación 29)"""
        D_CB_SW = self.calculate_D_CB_SW()
        D_T_SW = 0.0
        
        for (i, j), d_cb in D_CB_SW.items():
            f_ijs = self.flow_vars.get((i, j, node), 0)
            f_jis = self.flow_vars.get((j, i, node), 0)
            D_T_SW += d_cb * (f_ijs + f_jis)
        
        return D_T_SW
    
    def calculate_D_bar_SW(self, node):
        """Calcular D̄_s^SW (Ecuación 28)"""
        D_bar_SW = 0.0
        for u, v in self.G.edges():
            f_uvs = self.flow_vars.get((u, v, node), 0)
            f_vus = self.flow_vars.get((v, u, node), 0)
            lambda_uv = self.G[u][v]['lambda_val']
            tau_SW = self.G[u][v]['tau_SW']
            D_bar_SW += (f_uvs + f_vus) * lambda_uv * tau_SW
        return D_bar_SW
    
    def calculate_D_SW(self, node):
        """Calcular D_s^SW (Ecuación 27)"""
        return self.calculate_D_T_SW(node) - self.calculate_D_bar_SW(node)
    
    def calculate_all_indices(self):
        """Calcular todos los índices de confiabilidad"""
        results = {}
        
        for node in self.load_nodes:
            results[node] = {
                'N_RP': self.calculate_N_RP(node),
                'N_T': self.calculate_N_T(node),
                'N_SW': self.calculate_N_SW(node),
                'D_RP': self.calculate_D_RP(node),
                'D_T_SW': self.calculate_D_T_SW(node),
                'D_bar_SW': self.calculate_D_bar_SW(node),
                'D_SW': self.calculate_D_SW(node)
            }
        
        return results
    
    def load_customer_data(self, customer_data_file):
        """Cargar datos de clientes y cargas desde CSV"""
        try:
            if os.path.exists(customer_data_file):
                self.customer_df = pd.read_csv(customer_data_file)
                required_columns = ['Node', 'NC', 'L_MW']
                
                for col in required_columns:
                    if col not in self.customer_df.columns:
                        print(f"Advertencia: Columna {col} no encontrada en {customer_data_file}")
                        return
                
                # Crear diccionarios para acceso rápido
                self.customers_per_node = dict(zip(self.customer_df['Node'], self.customer_df['NC']))
                self.load_per_node = dict(zip(self.customer_df['Node'], self.customer_df['L_MW']))
                
                print(f"Datos de clientes cargados desde: {customer_data_file}")
                self.has_customer_data = True
            else:
                print(f"Archivo {customer_data_file} no encontrado. Se omitirán los índices adicionales.")
                self.has_customer_data = False
                
        except Exception as e:
            print(f"Error cargando datos de clientes: {str(e)}")
            self.has_customer_data = False
    
    def calculate_additional_indices(self):
        """Calcular índices adicionales CIF, CID, SAIFI, SAIDI, ASAI, EENS"""
        if not hasattr(self, 'has_customer_data') or not self.has_customer_data:
            return
        
        # Calcular CIF y CID por nodo
        self.additional_indices = {}
        
        for node in self.load_nodes:
            if node in self.customers_per_node:
                indices = self.reliability_indices[node]
                
                # CIF_s = N_s^RP + N_s^SW
                CIF_s = indices['N_RP'] + indices['N_SW']
                
                # CID_s = D_s^RP + D_s^SW
                CID_s = indices['D_RP'] + indices['D_SW']
                
                self.additional_indices[node] = {
                    'NC': self.customers_per_node[node],
                    'L_MW': self.load_per_node[node],
                    'CIF_s': CIF_s,
                    'CID_s': CID_s
                }
        
        # Calcular índices del sistema
        self.calculate_system_indices()
    
    def calculate_system_indices(self):
        """Calcular SAIFI, SAIDI, ASAI, EENS"""
        if not self.additional_indices:
            return
        
        # SAIFI y SAIDI
        numerator_saifi = 0.0
        numerator_saidi = 0.0
        denominator = 0.0
        
        for node, data in self.additional_indices.items():
            nc = data['NC']
            cif = data['CIF_s']
            cid = data['CID_s']
            
            numerator_saifi += nc * cif
            numerator_saidi += nc * cid
            denominator += nc
        
        self.SAIFI = numerator_saifi / denominator if denominator > 0 else 0.0
        self.SAIDI = numerator_saidi / denominator if denominator > 0 else 0.0
        
        # ASAI
        self.ASAI = 1 - (self.SAIDI / 8760)
        
        # EENS (con factor de carga μ = 1.0 por defecto)
        self.EENS = 0.0
        for node, data in self.additional_indices.items():
            cid = data['CID_s']
            load = data['L_MW']
            self.EENS += cid * load  # MWh/año
    
    def print_network_topology(self):
        """Mostrar topología de la red"""
        print(f"\n=== TOPOLOGÍA DE LA RED ===")
        print(f"Nodo raíz (subestación): {self.root}")
        print(f"Nodos de carga: {self.load_nodes}")
        print(f"Total de ramas: {len(self.G.edges())}")
        
        print(f"\nCaminos desde subestación:")
        for node in self.load_nodes:
            path_str = ' → '.join(map(str, self.paths[node]))
            print(f"  A nodo {node}: {path_str}")
    
    def print_detailed_results(self):
        """Mostrar resultados detallados finales"""
        self.print_network_topology()
        
        print(f"\n=== ÍNDICES DE CONFIABILIDAD POR NODO ===")
        print(f"{'Nodo':<6} {'N_RP':<8} {'N_T':<8} {'N_SW':<8} {'D_RP':<8} {'D_T,SW':<8} {'D̄_SW':<8} {'D_SW':<8}")
        print(f"{'':6} {'[int/año]':<8} {'[int/año]':<8} {'[int/año]':<8} {'[h/año]':<8} {'[h/año]':<8} {'[h/año]':<8} {'[h/año]':<8}")
        print("-" * 70)
        
        for node in sorted(self.load_nodes):
            indices = self.reliability_indices[node]
            print(f"{node:<6} {indices['N_RP']:<8.4f} {indices['N_T']:<8.4f} {indices['N_SW']:<8.4f} "
                  f"{indices['D_RP']:<8.4f} {indices['D_T_SW']:<8.4f} {indices['D_bar_SW']:<8.4f} {indices['D_SW']:<8.4f}")
        
        # Resumen estadístico
        print(f"\n=== RESUMEN ESTADÍSTICO ===")
        df_results = pd.DataFrame(self.reliability_indices).T
        
        print(f"\nPromedios:")
        for col in df_results.columns:
            print(f"  {col}: {df_results[col].mean():.4f}")
        
        print(f"\nNodos con mayor impacto:")
        print(f"  Mayor N_RP: Nodo {df_results['N_RP'].idxmax()} ({df_results['N_RP'].max():.4f})")
        print(f"  Mayor D_RP: Nodo {df_results['D_RP'].idxmax()} ({df_results['D_RP'].max():.4f})")
        print(f"  Mayor N_SW: Nodo {df_results['N_SW'].idxmax()} ({df_results['N_SW'].max():.4f})")
        print(f"  Mayor D_SW: Nodo {df_results['D_SW'].idxmax()} ({df_results['D_SW'].max():.4f})")
        
        # NUEVA SECCIÓN: ÍNDICES ADICIONALES
        self.print_additional_indices()
    
    def print_additional_indices(self):
        """Mostrar índices adicionales en cuadro separado"""
        if not hasattr(self, 'has_customer_data') or not self.has_customer_data:
            print(f"\n=== ÍNDICES ADICIONALES NO DISPONIBLES ===")
            print("No se encontraron datos de clientes en 6_NODOS.csv")
            return
        
        print(f"\n=== ÍNDICES ADICIONALES POR NODO ===")
        print(f"{'Nodo':<6} {'Clientes':<10} {'Carga':<10} {'CIF_s':<10} {'CID_s':<10}")
        print(f"{'':6} {'NC':<10} {'[MW]':<10} {'[int/año]':<10} {'[h/año]':<10}")
        print("-" * 56)
        
        for node in sorted(self.load_nodes):
            if node in self.additional_indices:
                data = self.additional_indices[node]
                print(f"{node:<6} {data['NC']:<10} {data['L_MW']:<10.2f} {data['CIF_s']:<10.4f} {data['CID_s']:<10.4f}")
        
        # Índices del sistema
        print(f"\n=== ÍNDICES DE CONFIABILIDAD DEL SISTEMA ===")
        print(f"SAIFI (System Average Interruption Frequency Index): {self.SAIFI:.6f} int/cliente-año")
        print(f"SAIDI (System Average Interruption Duration Index):  {self.SAIDI:.6f} h/cliente-año")
        print(f"ASAI  (Average Service Availability Index):          {self.ASAI:.8f} = {self.ASAI*100:.6f}%")
        print(f"EENS  (Expected Energy Not Served):                 {self.EENS:.6f} MWh/año")
        
        # Información adicional del sistema
        total_customers = sum(data['NC'] for data in self.additional_indices.values())
        total_load = sum(data['L_MW'] for data in self.additional_indices.values())
        
        print(f"\n=== INFORMACIÓN DEL SISTEMA ===")
        print(f"Total de clientes: {total_customers}")
        print(f"Carga total:       {total_load:.2f} MW")
        print(f"Disponibilidad:    {self.ASAI*100:.4f}%")


if __name__ == "__main__":
    try:
        # Ejecutar análisis con tu archivo DATA.csv y 6_NODOS.csv
        analyzer = RadialNetworkReliability()
        
        # Mostrar resultados detallados (incluye los índices adicionales)
        analyzer.print_detailed_results()
        
        print(f"\n=== ANÁLISIS COMPLETADO ===")
        print(f"Resultados calculados para {len(analyzer.load_nodes)} nodos de carga")
        
    except FileNotFoundError:
        print("ERROR: No se encontró el archivo DATA.csv")
        print("Por favor, crea tu archivo DATA.csv con las columnas:")
        print("u, v, lambda, Tao_RP, Tao_SW")
    except Exception as e:
        print(f"Error en la ejecución: {str(e)}")
        print("Verifique que el archivo DATA.csv tiene el formato correcto:")
