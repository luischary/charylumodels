from multiprocessing import Process
from typing import List
import time

import numpy as np


def run_pool_of_processes(
    estoque_processos: List[Process],
    num_workers: int,
    timeout_delaying_process: bool = False,
    process_timeout: int = 60,
):
    quantidade_estoque = 0
    quantidade_ativo = 0
    processos_ativos = []
    quantidade_estoque_original = len(estoque_processos)
    tempo_ultimo = time.time()

    while len(estoque_processos) > 0 or len(processos_ativos) > 0:
        matou_algum = True
        while matou_algum:  # itera para ir matando os caras que finalizaram
            matou_algum = False
            for i in range(len(processos_ativos)):
                if not processos_ativos[i].is_alive():
                    p = processos_ativos.pop(i)
                    p.terminate()
                    del p
                    tempo_ultimo = time.time()
                    break

        while len(processos_ativos) < num_workers and len(estoque_processos) > 0:
            indice = np.random.randint(len(estoque_processos))
            p = estoque_processos.pop(indice)
            p.start()
            processos_ativos.append(p)

        # isso aqui eh so para printar e ir acompanhando o quanto ja rodou
        if quantidade_estoque != len(estoque_processos) or quantidade_ativo != len(
            processos_ativos
        ):
            quantidade_estoque = len(estoque_processos)
            quantidade_ativo = len(processos_ativos)
            percentual_completo = (
                1
                - (quantidade_estoque + quantidade_ativo) / quantidade_estoque_original
            )
            percentual_completo *= 100
            print(
                f"Estoque: {len(estoque_processos)}\tAtivos: {len(processos_ativos)}\tPercent: {percentual_completo:3.4f}"
            )

            # quando a parada demora muito para processar vamos dar timeout
            if timeout_delaying_process:
                if time.time() - tempo_ultimo >= process_timeout:
                    if len(processos_ativos) > 0:
                        p = processos_ativos.pop(0)
                        p.terminate()
                        del p
                        tempo_ultimo = time.time()

        time.sleep(0.1)
