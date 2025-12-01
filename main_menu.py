import subprocess
import os

BASE_DIR = os.getcwd()

def run_gherkin():
    script = os.path.join(BASE_DIR, "src/application/use_cases/crew_gherkin.py")
    subprocess.run(["python", script], check=True)

def run_xunit():
    script = os.path.join(BASE_DIR, "src/application/use_cases/crew_xUnit.py")
    subprocess.run(["python", script], check=True)

def run_ieee():
    script = os.path.join(BASE_DIR, "src/application/use_cases/ieee_tests_plan_generator.py")
    subprocess.run(["python", script], check=True)
    
def menu():
    while True:
        print("\n=== MENU TESTE AI ===")
        print("1. Gerar Gherkin (.feature)")
        print("2. Gerar Steps (C# xUnit)")
        print("3. Gerar Plano de Testes IEEE")
        print("4. Sair")

        choice = input("")
        
        if choice == "1":
            try:
                run_gherkin()    
            except Exception as e:
                print(f"Erro: {e}")

        elif choice == "2":
            try:
                run_xunit()
            except Exception as e:
                print(f"Erro: {e}")

        elif choice == "3":
            try:
                run_ieee()
            except Exception as e:
                print(f"Erro: {e}")

        elif choice == "4":
            print("Saindo...")
            break

        else:
            print("Opção inválida!")


if __name__ == "__main__":
    menu()