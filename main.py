import os
from dotenv import load_dotenv

load_dotenv()


def main():
    print("=== Stem Agent ===\n")

    domain = input("Enter problem domain (e.g. 'Technical Research'): ").strip()

    if not domain:
        print("No domain provided. Exiting.")
        return

    print(f"\n[STEM] Starting transformation for domain: '{domain}'")
    print("[STEM] Observing domain...")
    print("[STEM] Synthesizing patterns...")
    print("[STEM] Generating specialized agent...\n")

    print(f"[AGENT] Ready to execute tasks in domain: '{domain}'")
    print("[AGENT] Enter a task (or 'exit' to quit):\n")

    while True:
        task = input("Task: ").strip()
        if task.lower() == "exit":
            break
        print(f"[AGENT] Processing: {task}\n")


if __name__ == "__main__":
    main()
