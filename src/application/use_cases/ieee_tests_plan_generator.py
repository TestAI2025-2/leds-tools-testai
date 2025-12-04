from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
from typing import Dict, List
from src.infrastructure.loaders.agent_loader import AgentLoader
from src.infrastructure.loaders.llm_loader import LLM_Loader
from src.infrastructure.loaders.task_yaml_loader import TaskLoader
from src.infrastructure.loaders.read_yaml import read_yaml_strings
import time
import os
import sys

load_dotenv()

def crew_ieee_to_gherkin(andes_content: str, strings: Dict[str, Dict]) -> str:
    
    # Gera um arquivo Gherkin (.feature) a partir de um plano de testes IEEE descrito em .andes.
    

    try:
        # Inicializa os LLMs
        llm_low_temp: LLM = LLM_Loader.load_from_params(temp=0.3)
        llm_high_temp: LLM = LLM_Loader.load_from_params(temp=0.7)

        # Carrega agentes e tarefas a partir dos YAMLs
        agents_dict = strings["agents"]
        tasks_dict = strings["tasks"]

        agents: List[Agent] = []
        tasks: List[Task] = []

        # Escritor do Gherkin em padrão IEEE
        writer_dict = agents_dict.get("ieee_writer")
        if not writer_dict:
            raise KeyError("Agente 'ieee_writer' não encontrado no YAML de agentes.")

        ieee_writer_agent: Agent = AgentLoader.load_agents(writer_dict, llm_high_temp)

        writer_task_dict = tasks_dict.get("ieee_writer_task")
        if not writer_task_dict:
            raise KeyError("Tarefa 'ieee_writer_task' não encontrada no YAML de tarefas.")

        writer_task_dict["description"] = writer_task_dict["description"].format(user_case=andes_content)
        task_ieee_writer: Task = TaskLoader.load_tasks(
            writer_task_dict,
            ieee_writer_agent,
            output_file="etapas_geracao/ieee_writer.feature"
        )

        # Review do gherkin
        reviewer_dict = agents_dict.get("ieee_reviewer")
        if not reviewer_dict:
            raise KeyError("Agente 'ieee_reviewer' não encontrado no YAML de agentes.")

        ieee_reviewer_agent: Agent = AgentLoader.load_agents(reviewer_dict, llm_low_temp)

        review_task_dict = tasks_dict.get("ieee_reviewer_task")
        if not review_task_dict:
            raise KeyError("Tarefa 'ieee_reviewer_task' não encontrada no YAML de tarefas.")

        review_task_dict["description"] = review_task_dict["description"].format(user_case=andes_content)
        task_ieee_review: Task = TaskLoader.load_tasks(
            review_task_dict,
            ieee_reviewer_agent,
            context=[task_ieee_writer],
            output_file="etapas_geracao/ieee_review.feature"
        )

        agents.extend([ieee_writer_agent, ieee_reviewer_agent])
        tasks.extend([task_ieee_writer, task_ieee_review])

        # Aprovação do manager
        manager_dict = agents_dict.get("manager_ieee")
        if not manager_dict:
            raise KeyError("Agente 'manager_ieee' não encontrado no YAML de agentes.")

        manager: Agent = AgentLoader.load_agents(manager_dict, llm_low_temp)

        manager_task_dict = tasks_dict.get("manager_ieee_task")
        if not manager_task_dict:
            raise KeyError("Tarefa 'manager_ieee_task' não encontrada no YAML de tarefas.")

        final_task: Task = TaskLoader.load_tasks(
            manager_task_dict,
            agent=manager,
            context=tasks,
            output_file="resposta/feature_from_ieee.feature",
        )

        crew: Crew = Crew(
            agents=agents + [manager],
            tasks=tasks + [final_task],
            max_rpm=10,
            output_log_file="crew_ieee_to_gherkin_log.txt",
            manager_llm=llm_low_temp,
            process=Process.sequential,
            verbose=True
        )

        resultado = crew.kickoff()
        return resultado.raw

    except Exception as e:
        print(f"[ERRO] Falha durante a geração do arquivo Gherkin: {e}")
        raise


if __name__ == "__main__":
    print("=== GERADOR DE ARQUIVO GHERKIN A PARTIR DE PLANO IEEE (.andes) ===")
    try:
        andes_name = input("Digite o nome do arquivo .andes (ex: test.andes): ").strip()
        
        andes_path = os.path.join("andes", andes_name)

        if not os.path.exists(andes_path):
            print(f"[ERRO] Arquivo '{andes_path}' não encontrado.")
            sys.exit(1)

        with open(andes_path, "r", encoding="utf-8") as file:
            andes_content = file.read()

        try:
            agents_dict, tasks_dict, outputs_dict = read_yaml_strings()
        except Exception as e:
            print(f"[ERRO] Falha ao ler YAMLs: {e}")
            sys.exit(1)

        strings = {"agents": agents_dict, "tasks": tasks_dict, "outputs": outputs_dict}

        start_time = time.time()
        resultado = crew_ieee_to_gherkin(andes_content, strings)
        end_time = time.time()

        print("\n=== GHERKIN GERADO COM SUCESSO ===\n")
        print(resultado)
        print(f"Tempo total de execução: {end_time - start_time:.2f}s")

    except KeyboardInterrupt:
        print("\n[AVISO] Execução interrompida pelo usuário.")
    except Exception as e:
        print(f"[ERRO FATAL] {e}")