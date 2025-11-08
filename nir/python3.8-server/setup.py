import os
import subprocess
import sys
import venv
from pathlib import Path

def create_env(env_path):
    if not env_path.exists():
        print("Создаю виртуальное окружение для coref_server...")
        builder = venv.EnvBuilder(with_pip=True)
        builder.create(env_path)

def install_requirements(env_path):
    pip = env_path / "bin" / "pip"
    print("Устанавливаю зависимости...")
    subprocess.check_call([str(pip), "install", "-r", "coref_server/requirements.txt"])

def start_server(env_path):
    python = env_path / "bin" / "python"
    server_script = Path("coref_server/coref_server.py")
    print("Запускаю coref_server на порту 8008...")
    subprocess.Popen([str(python), str(server_script)])

if __name__ == "__main__":
    env_dir = Path("coref_server/env38")
    create_env(env_dir)
    install_requirements(env_dir)
    start_server(env_dir)
    print("Сервер запущен. Проверка: http://127.0.0.1:8008/resolve")
