from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
from typing import Dict, List
from src.infrastructure.loaders.agent_loader import AgentLoader
from src.infrastructure.loaders.llm_loader import LLM_Loader
from src.infrastructure.loaders.task_yaml_loader import TaskLoader
from src.infrastructure.loaders.read_yaml import read_yaml_strings
import time
import os

load_dotenv()


def crew_ieee_test_plan(feature_content: str, strings: Dict[str, str]) -> str:
    """
    Gera um Plano de Teste IEEE a partir de um arquivo .feature.
    O plano segue o padrão IEEE 829 / ISO/IEC/IEEE 29119.
    """

    # Inicializa LLMs em diferentes temperaturas
    llm_low_temp: LLM = LLM_Loader.load_from_params()
    llm_high_temp: LLM = LLM_Loader.load_from_params(temp=0.6)

    # Carrega definições YAML (agentes e tarefas)
    agents_dict: Dict[str, dict] = strings["agents"]
    tasks_dict: Dict[str, dict] = strings["tasks"]

    agents: List[Agent] = []
    tasks: List[Task] = []

    # --- Etapa 1: geração do plano base IEEE ---
    writer_dict = agents_dict["ieee_writer"].copy()
    ieee_writer_agent: Agent = AgentLoader.load_agents(writer_dict, llm_high_temp)

    writer_task_dict = tasks_dict["ieee_writer_task"].copy()
    writer_task_dict["description"] = writer_task_dict["description"].format(feature_content=feature_content)
    task_ieee_writer: Task = TaskLoader.load_tasks(
        writer_task_dict,
        ieee_writer_agent,
        #output_file="etapas_geracao/ieee_writer.md"
    )

    # --- Etapa 2: revisão do plano ---
    reviewer_dict = agents_dict["ieee_reviewer"].copy()
    ieee_reviewer_agent: Agent = AgentLoader.load_agents(reviewer_dict, llm_low_temp)

    review_task_dict = tasks_dict["ieee_review_task"].copy()
    review_task_dict["description"] = review_task_dict["description"].format(feature_content=feature_content)
    task_ieee_review: Task = TaskLoader.load_tasks(
        review_task_dict,
        ieee_reviewer_agent,
        context=[task_ieee_writer],
        #output_file="etapas_geracao/ieee_review.md"
    )

    agents.extend([ieee_writer_agent, ieee_reviewer_agent])
    tasks.extend([task_ieee_writer, task_ieee_review])

    # --- Etapa 3: gerente final consolida ---
    manager: Agent = AgentLoader.load_agents(agents_dict["manager_ieee"], llm_low_temp)
    final_task: Task = TaskLoader.load_tasks(
        tasks_dict["manager_ieee_task"],
        agent=manager,
        context=tasks,
        output_file="resposta/plano_teste_ieee.md",
    )

    crew: Crew = Crew(
        agents=agents + [manager],
        tasks=tasks + [final_task],
        max_rpm=10,
        output_log_file="crew_ieee_log.txt",
        manager_llm=llm_low_temp,
        process=Process.sequential,
        verbose=True
    )

    resultado = crew.kickoff()
    return resultado.raw


if __name__ == "__main__":
    print("=== GERADOR DE PLANO DE TESTE IEEE ===")
    feature_name = input("Digite o nome do arquivo .feature (sem extensão): ").strip()
    feature_path = os.path.join("features", f"{feature_name}.feature")

    if not os.path.exists(feature_path):
        print(f"[ERRO] Arquivo '{feature_path}' não encontrado.")
        exit(1)

    with open(feature_path, "r", encoding="utf-8") as file:
        feature_content = file.read()

    agents_dict, tasks_dict, outputs_dict = read_yaml_strings()
    strings = {"agents": agents_dict, "tasks": tasks_dict, "outputs": outputs_dict}

    start_time = time.time()
    resultado = crew_ieee_test_plan(feature_content, strings)
    end_time = time.time()

    print(resultado)
    print(f"Tempo de execução: {end_time - start_time:.2f}s")
