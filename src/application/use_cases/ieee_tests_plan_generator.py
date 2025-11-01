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


def crew_ieee_to_gherkin(andes_content: str, strings: Dict[str, str]) -> str:
    """
    Gera um arquivo .feature (Gherkin) a partir de um plano de teste IEEE em .andes.
    """

    # Inicializa LLMs em diferentes temperaturas
    llm_low_temp: LLM = LLM_Loader.load_from_params()
    llm_high_temp: LLM = LLM_Loader.load_from_params(temp=0.6)

    # Carrega definições YAML (agentes e tarefas)
    agents_dict: Dict[str, dict] = strings["agents"]
    tasks_dict: Dict[str, dict] = strings["tasks"]

    agents: List[Agent] = []
    tasks: List[Task] = []

    # --- Etapa 1: geração do feature base a partir do plano IEEE ---
    writer_dict = agents_dict["gherkin_writer"].copy()
    gherkin_writer_agent: Agent = AgentLoader.load_agents(writer_dict, llm_high_temp)

    writer_task_dict = tasks_dict["gherkin_writer_task"].copy()
    writer_task_dict["description"] = writer_task_dict["description"].format(andes_content=andes_content)
    task_gherkin_writer: Task = TaskLoader.load_tasks(
        writer_task_dict,
        gherkin_writer_agent,
        output_file="etapas_geracao/gherkin_writer.feature"
    )

    # --- Etapa 2: revisão do feature ---
    reviewer_dict = agents_dict["gherkin_reviewer"].copy()
    gherkin_reviewer_agent: Agent = AgentLoader.load_agents(reviewer_dict, llm_low_temp)

    review_task_dict = tasks_dict["gherkin_review_task"].copy()
    review_task_dict["description"] = review_task_dict["description"].format(andes_content=andes_content)
    task_gherkin_review: Task = TaskLoader.load_tasks(
        review_task_dict,
        gherkin_reviewer_agent,
        context=[task_gherkin_writer],
        output_file="etapas_geracao/gherkin_review.feature"
    )

    agents.extend([gherkin_writer_agent, gherkin_reviewer_agent])
    tasks.extend([task_gherkin_writer, task_gherkin_review])

    # --- Etapa 3: gerente final consolida ---
    manager: Agent = AgentLoader.load_agents(agents_dict["manager_gherkin"], llm_low_temp)
    final_task: Task = TaskLoader.load_tasks(
        tasks_dict["manager_gherkin_task"],
        agent=manager,
        context=tasks,
        output_file="resposta/feature_from_ieee.feature",
    )

    crew: Crew = Crew(
        agents=agents + [manager],
        tasks=tasks + [final_task],
        max_rpm=10,
        output_log_file="crew_gherkin_log.txt",
        manager_llm=llm_low_temp,
        process=Process.sequential,
        verbose=True
    )

    resultado = crew.kickoff()
    return resultado.raw


if __name__ == "__main__":
    print("=== GERADOR DE FEATURE GHERKIN ===")
    andes_name = input("Digite o nome do arquivo .andes (sem extensão): ").strip()
    andes_path = os.path.join("resposta", f"{andes_name}.andes")

    if not os.path.exists(andes_path):
        print(f"[ERRO] Arquivo '{andes_path}' não encontrado.")
        exit(1)

    with open(andes_path, "r", encoding="utf-8") as file:
        andes_content = file.read()

    agents_dict, tasks_dict, outputs_dict = read_yaml_strings()
    strings = {"agents": agents_dict, "tasks": tasks_dict, "outputs": outputs_dict}

    start_time = time.time()
    resultado = crew_ieee_to_gherkin(andes_content, strings)
    end_time = time.time()

    print(resultado)
    print(f"Tempo de execução: {end_time - start_time:.2f}s")