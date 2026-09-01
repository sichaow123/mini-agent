from pathlib import Path

from dotenv import load_dotenv

from agent.agent import Agent

load_dotenv(override=False)


if __name__ == "__main__":
    print("Enter a question, press Enter to send. Type q to quit.\n")
    agent = Agent(str(Path.cwd()))

    try:
        while True:
            try:
                query = input("\033[36m>> \033[0m")
            except (EOFError, KeyboardInterrupt):
                break
            if query.strip().lower() in ("q", "exit", ""):
                break
            agent.react(query)
            print()
    finally:
        agent.clean_up()
